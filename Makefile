# Compiler
CC = nvcc

# Compiler for qSort
CC_QSORT = gcc

# Target executable
TARGET = cuda_bitonic

# Target executable for qSort
QSORT_TARGET = qSort

# File to store the last built version
LAST_VERSION_FILE = .last_version

# Default rule
all:
	@echo "Make Instructions:"
	@echo "  make <VERSION>         # Build the specified version (e.g., V0, V1, V2, qSort)"
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

qSort:
	$(MAKE) build VERSION=qSort

# Generic build rule
build:
	@if [ "$(VERSION)" = "qSort" ]; then \
		$(CC_QSORT) -o $(QSORT_TARGET)/$(QSORT_TARGET) qSort/qSort.c -lm; \
	else \
		$(CC) -o $(VERSION)/$(TARGET) main.cu $(VERSION)/bitonicSort.cu -I$(VERSION) -lm; \
	fi
	@echo "$(VERSION)" > $(LAST_VERSION_FILE)

# Run the program with q specified
run:
	@if [ ! -f $(LAST_VERSION_FILE) ]; then \
		echo "Error: No version built yet. Run 'make <VERSION>' first."; \
		exit 1; \
	fi
	@VERSION=$$(cat $(LAST_VERSION_FILE)); \
	if [ "$$VERSION" = "qSort" ]; then \
		echo "Running qSort with q=$(q)..."; \
		./$(QSORT_TARGET)/$(QSORT_TARGET) $(q); \
	else \
		echo "Running $$VERSION with q=$(q)..."; \
		./$$VERSION/$(TARGET) $(q); \
	fi

# Clean only the last built version and qSort directory if it exists
clean:
	@if [ ! -f $(LAST_VERSION_FILE) ]; then \
		echo "Error: No version built yet. Nothing to clean."; \
		exit 1; \
	fi
	@VERSION=$$(cat $(LAST_VERSION_FILE)); \
	if [ "$$VERSION" = "qSort" ]; then \
		rm -f $(QSORT_TARGET)/$(QSORT_TARGET); \
	else \
		rm -f $$VERSION/$(TARGET); \
	fi
	rm -f $(LAST_VERSION_FILE)

# Specify phony targets
.PHONY: all clean run V0 V1 V2 build qSort