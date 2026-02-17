GO      := go
BINARY  := bot
SRC     := bot.go

.PHONY: all run clean

all: $(BINARY)

$(BINARY): $(SRC)
	$(GO) build -o $(BINARY) $(SRC)

run: $(BINARY)
	./$(BINARY)

clean:
	rm -f $(BINARY)
