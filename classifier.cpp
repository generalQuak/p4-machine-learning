#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <set>

using namespace std;

class Classifier {
  private:
  map<string, int> class_counts;
  map<string, map<string, int>> word_counts;
  map<string, double> class_prob;
  map<string, map<string, double>> word_likely;

  public:
  void train(const string& train_filename) {

  }

  string predict(const string& post) {
    return "TBD";
  }
};

int main() {
  cout << "Hello World!\n";
}
