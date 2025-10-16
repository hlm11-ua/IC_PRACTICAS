# Nombre del compilador
CXX = g++
# Flags de compilación
#CXXFLAGS = -std=c++17 -Wall
#CXXFLAGS = -std=c++17 -Wall -O1
#CXXFLAGS = -std=c++17 -Wall -O2
#CXXFLAGS = -std=c++17 -Wall -O3
#CXXFLAGS = -std=c++17 -Wall -O2 -march=native
#CXXFLAGS = -std=c++17 -Wall -O3 -march=native
CXXFLAGS = -std=c++17 -Wall -O3 -funroll-loops
# Nombre del ejecutable
TARGET = GameOfLife
# Archivos fuente
SRC = GameOfLife.cpp
# Archivos objeto (reemplaza .cpp por .o)
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) -d 5 -fOut output.txt

clean:
	rm -f $(OBJ) $(TARGET) output.txt