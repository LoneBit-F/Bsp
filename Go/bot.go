package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// --- TUNING PARAMETER ---
const (
	ParticleCount     = 3000  // Anzahl Partikel pro Kanal
	ResampleThreshold = 0.5   // Ab wann Resampling (effektive Partikelanzahl)
	MinWeight         = 1e-20 // Untergrenze für Gewichte
	RandomInjectRate  = 0.05  // Prozentsatz zufälliger Partikel beim Resampling (Recovery)
	MaxAStarSteps     = 5000  // Max Schritte für A*
)

// --- DATENSTRUKTUREN ---
type Config struct {
	Width        int     `json:"width"`
	Height       int     `json:"height"`
	SignalRadius int     `json:"signal_radius"`
	SignalNoise  float64 `json:"signal_noise"`
	MaxGems      int     `json:"max_gems"`
}

type Input struct {
	Config      *Config   `json:"config,omitempty"`
	Tick        int       `json:"tick"`
	BotPos      [2]int    `json:"bot"`
	Walls       [][2]int  `json:"wall"`
	Floors      [][2]int  `json:"floor"`
	VisibleGems []GemInfo `json:"visible_gems"`
	Channels    []float64 `json:"channels"`
}

type GemInfo struct {
	Pos [2]int `json:"position"`
}

type Point struct {
	X, Y int
}

// --- ZUSTAND ---
type Particle struct {
	X, Y int
	W    float64
}

type PathNode struct {
	G, F      float64
	ParentIdx int
	HeapIdx   int
	LastSeen  int // Versionsnummer für Lazy Reset
	Closed    bool
	InOpen    bool
}

type BotState struct {
	GridWidth, GridHeight int
	Walls                 []bool
	Visited               []bool
	Particles             [][]Particle
	ResampleBuf           []Particle
	Nodes                 []PathNode
	SearchTick            int
	Config                Config
	Initialized           bool
	LogFile               *os.File // Handler für die Logdatei
}

// --- MAIN ---
func main() {
	reader := bufio.NewReaderSize(os.Stdin, 1024*1024)
	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	state := &BotState{}

	// Log-Datei öffnen (append mode, create if not exists)
	logFile, err := os.OpenFile("bot_debug.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		state.LogFile = logFile
		// Leere Zeile als Trenner bei neuem Run
		fmt.Fprintf(state.LogFile, "\n--- NEUER RUN START ---\n")
	} else {
		// Fallback auf Stderr falls Datei nicht geht (sollte aber klappen)
		state.LogFile = os.Stderr
	}
	defer state.LogFile.Close()

	scanner := bufio.NewScanner(reader)
	// Großer Buffer für Map-Daten
	buf := make([]byte, 4*1024*1024)
	scanner.Buffer(buf, 4*1024*1024)

	var input Input

	for scanner.Scan() {
		if err := json.Unmarshal(scanner.Bytes(), &input); err != nil {
			continue
		}

		if !state.Initialized && input.Config != nil {
			Initialize(state, *input.Config)
		}
		if !state.Initialized {
			fmt.Fprintln(writer, "WAIT")
			writer.Flush()
			continue
		}

		UpdateMap(state, &input)
		ProcessSignals(state, &input)
		move := Decide(state, &input)

		fmt.Fprintln(writer, move)
		writer.Flush()
	}
}

func Initialize(s *BotState, c Config) {
	s.Config = c
	s.GridWidth = c.Width
	s.GridHeight = c.Height
	size := c.Width * c.Height

	s.Walls = make([]bool, size)
	s.Visited = make([]bool, size)
	s.Particles = make([][]Particle, c.MaxGems)
	s.ResampleBuf = make([]Particle, ParticleCount)

	for i := 0; i < c.MaxGems; i++ {
		s.Particles[i] = make([]Particle, ParticleCount)
		ScatterParticlesUniform(s, i)
	}

	s.Nodes = make([]PathNode, size)
	s.Initialized = true

	if s.LogFile != nil {
		fmt.Fprintf(s.LogFile, "Init: Grid %dx%d, MaxGems %d, SignalRadius %d\n", c.Width, c.Height, c.MaxGems, c.SignalRadius)
	}
}

func ScatterParticlesUniform(s *BotState, channelIdx int) {
	particles := s.Particles[channelIdx]
	w := 1.0 / float64(ParticleCount)
	idx := 0
	// Deterministischer Scatter über das Grid
	for i := 0; i < ParticleCount; i++ {
		r := (i * 9973) % (s.GridWidth * s.GridHeight)
		x := r % s.GridWidth
		y := r / s.GridWidth
		particles[idx] = Particle{X: x, Y: y, W: w}
		idx++
	}
}

func UpdateMap(s *BotState, in *Input) {
	idx := in.BotPos[1]*s.GridWidth + in.BotPos[0]
	if idx >= 0 && idx < len(s.Visited) {
		s.Visited[idx] = true
	}
	for _, w := range in.Walls {
		wIdx := w[1]*s.GridWidth + w[0]
		if wIdx >= 0 && wIdx < len(s.Walls) {
			s.Walls[wIdx] = true
		}
	}
}

// --- MCL (MONTE CARLO LOCALIZATION) ---
func ProcessSignals(s *BotState, in *Input) {
	botPos := Point{in.BotPos[0], in.BotPos[1]}
	sigma := s.Config.SignalNoise
	if sigma <= 0 {
		sigma = 0.1
	}
	var2 := 2.0 * sigma * sigma
	normFactor := 1.0 / (math.Sqrt(2.0*math.Pi) * sigma)

	for cIdx, signal := range in.Channels {
		if cIdx >= len(s.Particles) {
			break
		}

		if signal == 0 {
			ScatterParticlesUniform(s, cIdx)
			continue
		}

		particles := s.Particles[cIdx]
		var totalWeight float64 = 0
		var sumSqWeights float64 = 0

		// 1. Gewichtung
		for i := range particles {
			p := &particles[i]
			gridIdx := p.Y*s.GridWidth + p.X

			// Partikel in Wand -> Gewicht 0
			if s.Walls[gridIdx] {
				p.W = 0
				continue
			}

			dx := float64(p.X - botPos.X)
			dy := float64(p.Y - botPos.Y)
			dist := math.Sqrt(dx*dx + dy*dy)

			term := dist / float64(s.Config.SignalRadius)
			expectedSignal := 1.0 / (1.0 + term*term)

			diff := signal - expectedSignal
			likelihood := normFactor * math.Exp(-(diff*diff)/var2)

			p.W *= likelihood
			totalWeight += p.W
		}

		// 2. Normalisierung & Deprivation Check
		if totalWeight > MinWeight {
			for i := range particles {
				particles[i].W /= totalWeight
				sumSqWeights += particles[i].W * particles[i].W
			}
		} else {
			// Signal passt absolut nicht zu den Partikeln -> Reset
			// LOG: Deprivation
			// fmt.Fprintf(s.LogFile, "MCL Channel %d Reset (Deprivation)\n", cIdx)
			ScatterParticlesUniform(s, cIdx)
			continue
		}

		// 3. Resampling
		nEff := 1.0 / sumSqWeights
		if nEff < float64(ParticleCount)*ResampleThreshold {
			SystematicResampling(s, cIdx)
		}
	}
}

func SystematicResampling(s *BotState, cIdx int) {
	oldParticles := s.Particles[cIdx]
	newParticles := s.ResampleBuf
	step := 1.0 / float64(ParticleCount)
	r := 0.5 * step
	c := oldParticles[0].W
	i := 0

	for m := 0; m < ParticleCount; m++ {
		// Random Injection: Ein kleiner Teil der Partikel wird zufällig verteilt,
		// um sich von falschen Konvergenzen zu erholen.
		if m < int(float64(ParticleCount)*RandomInjectRate) {
			randPos := (m * 12345) % (s.GridWidth * s.GridHeight) // Simple deterministic random
			newParticles[m] = Particle{
				X: randPos % s.GridWidth,
				Y: randPos / s.GridWidth,
				W: step,
			}
			continue
		}

		u := r + float64(m)*step
		for u > c && i < ParticleCount-1 {
			i++
			c += oldParticles[i].W
		}
		newParticles[m] = oldParticles[i]
		newParticles[m].W = step

		// Jitter
		if m%5 == 0 {
			jitterMove(s, &newParticles[m])
		}
	}
	copy(s.Particles[cIdx], newParticles)
}

func jitterMove(s *BotState, p *Particle) {
	moves := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	mv := moves[(p.X+p.Y)%4]
	nx, ny := p.X+mv[0], p.Y+mv[1]
	if nx >= 0 && nx < s.GridWidth && ny >= 0 && ny < s.GridHeight {
		if !s.Walls[ny*s.GridWidth+nx] {
			p.X = nx
			p.Y = ny
		}
	}
}

// --- ENTSCHEIDUNG ---
func Decide(s *BotState, in *Input) string {
	myPos := Point{in.BotPos[0], in.BotPos[1]}

	// LOG: Header für diesen Tick
	if s.LogFile != nil {
		fmt.Fprintf(s.LogFile, "[TICK %d] Pos: %v | Channels: %v\n", in.Tick, myPos, in.Channels)
	}

	// 1. Sichtbare Gems (höchste Prio)
	for _, gem := range in.VisibleGems {
		target := Point{gem.Pos[0], gem.Pos[1]}
		if s.LogFile != nil {
			fmt.Fprintf(s.LogFile, "  >> STRATEGIE: SICHTBARES GEM bei %v\n", target)
		}
		move := GetNextStep(s, myPos, target)
		if move != "WAIT" {
			return move
		}
		if s.LogFile != nil {
			fmt.Fprintf(s.LogFile, "     ! Weg blockiert.\n")
		}
	}

	// 2. MCL Ziele
	bestTarget := Point{-1, -1}
	maxConf := 0.0
	bestChannel := -1

	for i := 0; i < len(s.Particles); i++ {
		if in.Channels[i] == 0 {
			continue
		}

		center, conf := CalculateCentroid(s.Particles[i])
		// Nur wenn wir halbwegs sicher sind (conf > 0.01 z.B.)
		if center.X != -1 && conf > 0.01 {
			// SnapToFloor mit erweitertem Radius
			validTarget := SnapToFloorBFS(s, center)
			if validTarget.X != -1 {
				if conf > maxConf {
					maxConf = conf
					bestTarget = validTarget
					bestChannel = i
				}
			}
		}
	}

	if bestTarget.X != -1 {
		if s.LogFile != nil {
			fmt.Fprintf(s.LogFile, "  >> STRATEGIE: MCL (Kanal %d) -> Ziel %v (Conf: %.4f)\n", bestChannel, bestTarget, maxConf)
		}
		move := GetNextStep(s, myPos, bestTarget)
		if move != "WAIT" {
			return move
		}
		if s.LogFile != nil {
			fmt.Fprintf(s.LogFile, "     ! Weg blockiert.\n")
		}
	} else {
		if s.LogFile != nil {
			fmt.Fprintf(s.LogFile, "  >> INFO: Kein starkes MCL-Signal.\n")
		}
	}

	// 3. Exploration
	if s.LogFile != nil {
		fmt.Fprintf(s.LogFile, "  >> STRATEGIE: EXPLORATION\n")
	}
	move := ExploreBFS(s, myPos)
	if move != "WAIT" {
		return move
	}

	// 4. Panic Random Walk
	if s.LogFile != nil {
		fmt.Fprintf(s.LogFile, "  >> STRATEGIE: PANIC / RANDOM WALK\n")
	}
	return RandomValidMove(s, myPos)
}

// Sucht den nächsten freien Boden in der Nähe von p (BFS)
func SnapToFloorBFS(s *BotState, start Point) Point {
	startIdx := start.Y*s.GridWidth + start.X
	if startIdx >= 0 && startIdx < len(s.Walls) && !s.Walls[startIdx] {
		return start
	}

	queue := []Point{start}
	visited := make(map[Point]bool)
	visited[start] = true

	// Begrenzte Suche (z.B. max 5 Schritte Radius)
	steps := 0

	for len(queue) > 0 {
		if steps > 50 {
			break
		} // Max Checks
		steps++

		curr := queue[0]
		queue = queue[1:]

		idx := curr.Y*s.GridWidth + curr.X
		if idx >= 0 && idx < len(s.Walls) && !s.Walls[idx] {
			return curr
		}

		dirs := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
		for _, d := range dirs {
			nx, ny := curr.X+d[0], curr.Y+d[1]
			np := Point{nx, ny}
			if nx >= 0 && nx < s.GridWidth && ny >= 0 && ny < s.GridHeight && !visited[np] {
				visited[np] = true
				queue = append(queue, np)
			}
		}
	}
	return Point{-1, -1}
}

func CalculateCentroid(particles []Particle) (Point, float64) {
	var xSum, ySum, wSum float64
	maxW := 0.0
	for _, p := range particles {
		xSum += float64(p.X) * p.W
		ySum += float64(p.Y) * p.W
		wSum += p.W
		if p.W > maxW {
			maxW = p.W
		}
	}
	if wSum == 0 {
		return Point{-1, -1}, 0
	}
	return Point{
		X: int(math.Round(xSum / wSum)),
		Y: int(math.Round(ySum / wSum)),
	}, maxW // Nutzen maxW als Konfidenz-Metrik
}

// --- PFADFINDUNG & UTILS ---
func RandomValidMove(s *BotState, p Point) string {
	dirs := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
	perm := rand.Perm(4)
	for _, i := range perm {
		d := dirs[i]
		nx, ny := p.X+d[0], p.Y+d[1]
		if nx >= 0 && nx < s.GridWidth && ny >= 0 && ny < s.GridHeight {
			if !s.Walls[ny*s.GridWidth+nx] {
				return Direction(p, Point{nx, ny})
			}
		}
	}
	return "WAIT"
}

func GetNextStep(s *BotState, start, goal Point) string {
	path := AStar(s, start, goal)
	if len(path) > 0 {
		return Direction(start, path[0])
	}
	return "WAIT"
}

// A* Implementierung
type NodeHeap struct {
	Items []int
	State *BotState
}

func (h *NodeHeap) Push(nodeIdx int) {
	h.Items = append(h.Items, nodeIdx)
	h.State.Nodes[nodeIdx].HeapIdx = len(h.Items) - 1
	h.up(len(h.Items) - 1)
}

func (h *NodeHeap) Pop() int {
	n := len(h.Items) - 1
	h.swap(0, n)
	h.down(0, n)
	x := h.Items[n]
	h.Items = h.Items[0:n]
	h.State.Nodes[x].HeapIdx = -1
	return x
}

func (h *NodeHeap) Update(nodeIdx int) {
	idx := h.State.Nodes[nodeIdx].HeapIdx
	if idx != -1 {
		h.up(idx)
	}
}

func (h *NodeHeap) swap(i, j int) {
	h.Items[i], h.Items[j] = h.Items[j], h.Items[i]
	h.State.Nodes[h.Items[i]].HeapIdx = i
	h.State.Nodes[h.Items[j]].HeapIdx = j
}

func (h *NodeHeap) up(j int) {
	for {
		i := (j - 1) / 2
		if i == j || h.less(i, j) {
			break
		}
		h.swap(i, j)
		j = i
	}
}

func (h *NodeHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		j2 := j1 + 1
		if j2 < n && h.less(j2, j1) {
			j = j2
		}
		if !h.less(j, i) {
			break
		}
		h.swap(i, j)
		i = j
	}
}

func (h *NodeHeap) less(i, j int) bool {
	ni, nj := h.Items[i], h.Items[j]
	return h.State.Nodes[ni].F < h.State.Nodes[nj].F
}

// Initialisiert Knoten bei Bedarf (Lazy Reset)
func (s *BotState) getNode(idx int) *PathNode {
	n := &s.Nodes[idx]
	if n.LastSeen < s.SearchTick {
		n.G = math.Inf(1)
		n.F = math.Inf(1)
		n.ParentIdx = -1
		n.Closed = false
		n.InOpen = false
		n.LastSeen = s.SearchTick
	}
	return n
}

func AStar(s *BotState, start, goal Point) []Point {
	s.SearchTick++
	startIdx := start.Y*s.GridWidth + start.X
	goalIdx := goal.Y*s.GridWidth + goal.X

	startNode := s.getNode(startIdx)
	startNode.G = 0
	startNode.F = heuristic(start, goal)
	startNode.InOpen = true

	openSet := NodeHeap{Items: make([]int, 0, 100), State: s}
	openSet.Push(startIdx)

	loopCount := 0
	for len(openSet.Items) > 0 {
		loopCount++
		if loopCount > MaxAStarSteps {
			break
		}

		currentIdx := openSet.Pop()
		currNode := s.Nodes[currentIdx] // Kopie ist ok, wir ändern nur Closed
		s.Nodes[currentIdx].Closed = true
		s.Nodes[currentIdx].InOpen = false

		if currentIdx == goalIdx {
			return reconstructPath(s, currentIdx)
		}

		cy := currentIdx / s.GridWidth
		cx := currentIdx % s.GridWidth

		dirs := [][2]int{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}
		for _, d := range dirs {
			nx, ny := cx+d[0], cy+d[1]
			if nx < 0 || nx >= s.GridWidth || ny < 0 || ny >= s.GridHeight {
				continue
			}
			nIdx := ny*s.GridWidth + nx

			// Wände blockieren
			if s.Walls[nIdx] {
				continue
			}

			// Nachbar holen (automatisch resettet wenn nötig)
			neighbor := s.getNode(nIdx)

			if neighbor.Closed {
				continue
			}

			tentativeG := currNode.G + 1.0

			if tentativeG < neighbor.G {
				neighbor.ParentIdx = currentIdx
				neighbor.G = tentativeG
				neighbor.F = tentativeG + heuristic(Point{nx, ny}, goal)

				if !neighbor.InOpen {
					neighbor.InOpen = true
					openSet.Push(nIdx)
				} else {
					openSet.Update(nIdx)
				}
			}
		}
	}
	return nil
}

func reconstructPath(s *BotState, currentIdx int) []Point {
	var path []Point
	curr := currentIdx
	for curr != -1 {
		cy := curr / s.GridWidth
		cx := curr % s.GridWidth
		path = append(path, Point{cx, cy})
		// Hier müssen wir aufpassen: getNode resettet, aber wir lesen nur.
		// Wenn ParentIdx aus dem aktuellen Tick stammt, ist er gültig.
		// Da wir rückwärts gehen vom Ziel (das im aktuellen Tick gefunden wurde),
		// sind alle Parents auch aktuell.
		curr = s.Nodes[curr].ParentIdx
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	if len(path) > 0 {
		path = path[1:]
	}
	return path
}

func heuristic(a, b Point) float64 {
	return math.Abs(float64(a.X-b.X)) + math.Abs(float64(a.Y-b.Y))
}

func ExploreBFS(s *BotState, start Point) string {
	startIdx := start.Y*s.GridWidth + start.X
	queue := []int{startIdx}

	visited := make(map[int]bool)
	visited[startIdx] = true
	parents := make(map[int]int)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		// Ziel: Unbekanntes Feld
		if !s.Visited[curr] && !s.Walls[curr] {
			temp := curr
			for parents[temp] != startIdx {
				temp = parents[temp]
			}
			ty := temp / s.GridWidth
			tx := temp % s.GridWidth
			return Direction(start, Point{tx, ty})
		}

		cy := curr / s.GridWidth
		cx := curr % s.GridWidth
		dirs := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}

		for _, d := range dirs {
			nx, ny := cx+d[0], cy+d[1]
			if nx >= 0 && nx < s.GridWidth && ny >= 0 && ny < s.GridHeight {
				nIdx := ny*s.GridWidth + nx
				if !s.Walls[nIdx] && !visited[nIdx] {
					visited[nIdx] = true
					parents[nIdx] = curr
					queue = append(queue, nIdx)
				}
			}
		}
	}
	return "WAIT"
}

func Direction(from, to Point) string {
	if to.Y < from.Y {
		return "N"
	}
	if to.Y > from.Y {
		return "S"
	}
	if to.X > from.X {
		return "E"
	}
	if to.X < from.X {
		return "W"
	}
	return "WAIT"
}
