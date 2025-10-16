#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace std;

const int ITER_BY_SIZE = 100;

enum CellStates { DEATH, LIVE };

class GameOfLife {
    private:
        char liveChar = '1';
        char deathChar = '0';
        int gen;
        int totalCells;
        int liveCells;
        int X_DIM;
        int Y_DIM;
        int MAX_GENS = 1000;
        vector<vector<CellStates>> grid;
        vector<vector<int>> neighbors;

    public:
        GameOfLife(int x = 50, int y = 50);
        void initializeGrid();
        void initializeGridFromFile(string filename);
        int nextLiveCells(int i, int j);
        void nextGen();
        int countLiveCells();
        void start(string filename);
        void start(int Runs, string filename);
        double neighborsAvg();
        double neighborsDensity();
        double stability(double previousDensity);
};

GameOfLife::GameOfLife(int x, int y) {
    X_DIM = x;
    Y_DIM = y;
    totalCells = X_DIM * Y_DIM;
    neighbors.resize(x, vector<int>(y, 0));
    gen = 0;
    liveCells = 0;
}

double GameOfLife::neighborsAvg() {
    int totalNeighbors = 0;
    for (int i = 0; i < X_DIM; i++)
        for (int j = 0; j < Y_DIM; j++)
            totalNeighbors += neighbors[i][j];

    return (double)totalNeighbors / totalCells;
}

double GameOfLife::neighborsDensity() {
    double difSquaredSum = 0.0;
    double avg = neighborsAvg();
    for (int i = 0; i < X_DIM; i++) {
        for (int j = 0; j < Y_DIM; j++) {
            int n = neighbors[i][j];
            difSquaredSum += (n - avg) * (n - avg);
        }
    }
    return difSquaredSum / totalCells;
}

double GameOfLife::stability(double previousDensity) {
    return fabs(neighborsDensity() - previousDensity);
}

// void GameOfLife::initializeGridFromFile(string filename) {
//     ifstream inFile(filename);
//     if (!inFile.is_open()) {
//         cerr << "Error opening file: " << filename << endl;
//         return;
//     }
//     grid = vector<vector<CellStates>>(X_DIM);
//     for (int i = 0; i < X_DIM; i++) {
//         grid[i] = vector<CellStates>(Y_DIM);
//         for (int j = 0; j < Y_DIM; j++) {
//             char cell;
//             inFile >> cell;
//             if (cell == liveChar)
//                 grid[i][j] = LIVE;
//             else
//                 grid[i][j] = DEATH;
//         }
//     }
// }

void GameOfLife::initializeGrid() {
    srand(time(0));
    grid = vector<vector<CellStates>>(X_DIM);
    for (int i = 0; i < X_DIM; i++) {
        grid[i] = vector<CellStates>(Y_DIM);
        for (int j = 0; j < Y_DIM; j++)
            grid[i][j] = CellStates(rand() % 2);
    }
}

int GameOfLife::nextLiveCells(int i, int j) {
    int lives = 0;
    for (int k = -1; k <= 1; k++) {
        for (int h = -1; h <= 1; h++) {
            if (k == 0 && h == 0)
                continue;

            int x = (k + i + X_DIM) % X_DIM;
            int y = (h + j + Y_DIM) % Y_DIM;
            lives += (int)grid[x][y];
        }
    }
    neighbors[i][j] = lives;
    return lives;
}

void GameOfLife::nextGen() {
    vector<vector<CellStates>> tmpGrid(X_DIM);
    for (int i = 0; i < X_DIM; i++) {
        tmpGrid[i] = vector<CellStates>(Y_DIM);
        for (int j = 0; j < Y_DIM; j++) {
            int n = nextLiveCells(i, j);

            if ((grid[i][j] == LIVE && (n == 2 || n == 3)) ||
                (grid[i][j] == DEATH && n == 3))
                tmpGrid[i][j] = LIVE;
            else
                tmpGrid[i][j] = DEATH;
        }
    }
    grid = tmpGrid;
}

int GameOfLife::countLiveCells() {
    int liveCells = 0;
    for (int i = 0; i < X_DIM; i++)
        for (int j = 0; j < Y_DIM; j++)
            if (grid[i][j] == LIVE)
                liveCells++;
    return liveCells;
}

void GameOfLife::start(int runs, string fich) {
    double DensityVal = neighborsDensity();

    while (gen < runs) {
        nextGen();
        gen++;
        liveCells = countLiveCells();
        if (!fich.empty()) {
            ofstream outFile(fich, std::ios::out | std::ios::app);
            if (!outFile.is_open()) {
                cerr << "Error opening file: " << fich << endl;
                return;
            }
            outFile << "Gen: " << gen << " - Live cells: " << liveCells << "/"
                    << totalCells << " - Avg: " << neighborsAvg()
                    << " - Stability: " << stability(DensityVal)
                    << " - Density: " << neighborsDensity() << "\n";
        }
        DensityVal = neighborsDensity();
    }
}

void GameOfLife::start(string fich) { start(MAX_GENS, fich); }

/// ----------------------
/// Argument checker igual
/// ----------------------
bool argumentCheck(
    int argc, char *argv[], int &duplications, string &outFileName
) {
    bool outFile, dups;
    outFile = dups = false;

    if (argc != 5) {
        cerr << "Usage: GameOfLife -d <duplications> -fOut <FILENAME>" << endl;
        return false;
    }
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-d" && i + 1 < argc && !dups) {
            duplications = stoi(argv[i + 1]);
            dups = true;
            i++;
        } else if (arg == "-fOut" && i + 1 < argc && !outFile) {
            outFileName = argv[i + 1];
            outFile = true;
            i++;
        } else {
            cerr << "Usage: GameOfLife -d <duplications> -fOut <FILENAME>" << endl;
            return false;
        } 
    }

    if (outFileName.empty()) {
        cerr << "Usage: GameOfLife -d <duplications> -fOut <FILENAME>" << endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    string outFileName = "";
    float avg_time = 0; 
    int X_DIM = 50, Y_DIM = 50, duplications = 20;

    if (!argumentCheck(argc, argv, duplications, outFileName)) return 1;
    
    for (int i = 0; i < duplications; i++) {
        GameOfLife game(X_DIM, Y_DIM);
        for (int k = 0; k < ITER_BY_SIZE; k++) {
            auto start = clock();
            game.initializeGrid();
            game.start(outFileName);
            auto end = clock();

            avg_time += 1000.0 * (end - start) / CLOCKS_PER_SEC;
        }

        cout << "Size: " << X_DIM << "x" << Y_DIM << "\t|\tTime: "
                << avg_time / ITER_BY_SIZE << " ms" << endl;
        X_DIM *= 2; Y_DIM *= 2;
    }

    avg_time /= (duplications * ITER_BY_SIZE);
    cout << "Average time: " << avg_time << " ms" << endl;
    return 0;
}