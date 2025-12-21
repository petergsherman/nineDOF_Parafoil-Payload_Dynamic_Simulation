CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native
LDFLAGS = 

# Source files
SOURCES = nineDOF_Main.cpp nineDOF_Plant.cpp
HEADERS = nineDOF_Parameters.h nineDOF_Transform.h nineDOF_Plant.h

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = nineDOF_sim

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) trajectory.csv

# Run the simulation
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: clean $(TARGET)

.PHONY: all clean run debug