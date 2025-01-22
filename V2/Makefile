# Compiler
CC = nvcc

# Source files
SRCS = main.cu bitonicSort.cu 

# Object files (replace .cu with .o)
OBJS = $(SRCS:.cu=.o)

# Target executable
TARGET = cuda_bitonic

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) -lm

# Compile source files to object files
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program with LD_LIBRARY_PATH set
run: $(TARGET)
	./$(TARGET)

# Specify phony targets
.PHONY: all clean run