#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include "csvstream.hpp"

using namespace std;

/**Before the classifier can make predictions, it needs to be trained on a set of previously 
 * labeled Piazza posts (e.g. train_small.csv or w16_projects_exam.csv). 
 * Your application should process each post in the training set, and record the following information:

The total number of posts in the entire training set.
The number of unique words in the entire training set. (The vocabulary size.)
For each word 'w', the number of posts in the entire training set that contain 'w'.
For each label 'C', the number of posts with that label.
For each label 'C' and word, the number of posts with label that contain 'w'. 


csv examples https://github.com/awdeorio/csvstream
 */


/**
 * float: Requires 4 bytes in memory, with a precision of up to 6 digits.
 * double: Requires 8 bytes in memory, with a precision of up to 15 digits.
 * long double: Requires 10 bytes in memory, with a precision of up to 19 digits
 * std::log()
 */

/**
 * Purpose, etc
 */
class Classifier {
  private:
  map<string, int> class_counts;
  map<string, map<string, int>> word_counts;
  map<string, double> class_prob;
  map<string, map<string, double>> word_likely;

  public:
  Classifier() {
  }

  void train(istream &stream) {  
    string line;
    while (getline(stream, line)) {
      istringstream ss(line);
      string label, post;

      // Parse labels and content from a line
      getline(ss, label, ",");
      getline(ss, post);

      class_counts[label]++;
    }

  }

  string predict(const string& post) {
    return "TBD";
  }
};

/**
 * EFFECT: Utilizes a vector of probabilities to calculate and return double probability
 */
double calculate_log_probability(const vector<double>& probabilities) {
    double log_sum = 0.0;
    for (const auto& p : probabilities) {
        log_sum += log(p);
    }
    return log_sum;
}

/**
 * EFFECT: Returns the probability of 'C' out of 'T' sets ln 
 */
double probability_of(int numC, int numT) {
  return log(numC/numT);
}

/**
 * EFFECT: Returns the probability of a word 'w' in label 'C' 
 * If 'w' doesn't occur in 'C': ln P(w|C) = ln (num 'w'/num sets T)
 * If 'w' doesn't occur ANYWHERE: ln P(w|C) = ln(1/num sets T)
 * Default: ln P(w|C) = ln (num sets 'C' & 'w' / num sets  'C')
 */
double word_prob(int numW, int numT) {
  if (false){
    //to be reworked then implemented
  } else if (false) { return log(1/numT); } //if 'w' doesn't occur in sets T
  
  return log(numW/numT);
}

/**
 *  EFFECTS: Return a set of unique whitespace delimited words
*/ 
set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}


////////////////////////////////// MAIN FUNCTION ///////////////////////////

int main(int argc, char **argv) {
  
  // Opening file streams
  ifstream trainFile(argv[1]);
  ifstream testFile; //initialize testFile
  if(argc == 3){ testFile.open(argv[2]); } //opens testFile

  // ERROR CHECKING:
  if (argc != 2 && argc != 3){
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }

  // File streams check
  if (!trainFile.is_open()){
    cout << "Error opening file: " << argv[1] << endl;
    return 1;
  }
  
  if (argc == 3 && !testFile.is_open()) { 
    cout << "Error opening file: " << argv[2] << endl;
    return 1;
  } 



  return 0;
}