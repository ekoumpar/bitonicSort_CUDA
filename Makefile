# Compiler
CC = nvcc

# Target executable
TARGET = cuda_bitonic

# File to store the last built version
LAST_VERSION_FILE = .last_version

# Default rule
all:
	@echo "Make Instructions:"
	@echo "  make <VERSION>         # Build the specified version (e.g., V0, V1, V2)"
	@echo "  make run q=<Value>     # Run the last built version with q=<Value>"
	@echo "Example:"
	@echo "  make V0"
	@echo "  make run q=20"

# Build rule for each version
V0:
	$(MAKE) build VERSION=V0

V1:
	$(MAKE) build VERSION=V1

V2:
	$(MAKE) build VERSION=V2

# Generic build rule
build:
	$(CC) -o $(VERSION)/$(TARGET) $(VERSION)/main.cu $(VERSION)/bitonicSort.cu -I$(VERSION) -lm
	@echo "$(VERSION)" > $(LAST_VERSION_FILE)

# Run the program with q specified
run:
	@if [ ! -f $(LAST_VERSION_FILE) ]; then \
		echo "Error: No version built yet. Run 'make <VERSION>' first."; \
		exit 1; \
	fi
	@VERSION=$$(cat $(LAST_VERSION_FILE)); \
	echo "Running $$VERSION with q=$(q)..."; \
	./$$VERSION/$(TARGET) $(q)

# Clean only the last built version
clean:
	@if [ ! -f $(LAST_VERSION_FILE) ]; then \
		echo "Error: No version built yet. Nothing to clean."; \
		exit 1; \
	fi
	@VERSION=$$(cat $(LAST_VERSION_FILE)); \
	rm -f $$VERSION/$(TARGET); \
	rm -f $(LAST_VERSION_FILE)

# Specify phony targets
.PHONY: all clean run V0 V1 V2 build