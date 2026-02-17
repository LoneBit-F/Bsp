package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type Config struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}
type Tick struct {
	Config Config `json:"config"`
}

func main() {
	// Deterministic RNG (match other baselines)
	r := rand.New(rand.NewSource(1))

	// Unbuffered-ish writes by printing with newline
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	sc := bufio.NewScanner(os.Stdin)
	// Allow long JSON lines
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 1024*1024)

	firstTick := true
	moves := []string{"N", "S", "E", "W"}

	for sc.Scan() {
		line := sc.Bytes()

		if firstTick {
			var t Tick
			_ = json.Unmarshal(line, &t) // ignore errors; still emit a move
			fmt.Fprintf(os.Stderr, "Random walker (Go) launching on a %dx%d map\n", t.Config.Width, t.Config.Height)
			firstTick = false
		}

		fmt.Fprintln(out, moves[r.Intn(len(moves))])
		out.Flush()
	}

	if err := sc.Err(); err != nil {
		// If stdin closes with an error, log to stderr (won't affect stdout protocol)
		fmt.Fprintln(os.Stderr, "stdin error:", err, "at", time.Now().Format(time.RFC3339))
	}
}
