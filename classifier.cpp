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
 * taged Piazza posts (e.g. train_small.csv or w16_projects_exam.csv). 
 * Your application should process each post in the training set, and record the following information:

The total number of posts in the entire training set.
The number of unique words in the entire training set. (The vocabulary size.)
For each word 'w', the number of posts in the entire training set that contain 'w'.
For each tag 'C', the number of posts with that tag.
For each tag 'C' and word, the number of posts with tag that contain 'w'. 


csv examples https://github.com/awdeorio/csvstream
 */


/**
 * float: Requires 4 bytes in memory, with a precision of up to 6 digits.
 * double: Requires 8 bytes in memory, with a precision of up to 15 digits.
 * long double: Requires 10 bytes in memory, with a precision of up to 19 digits
 * std::log()
 */

/**
 * BRIEF: Class that holds necessary data for calculating the probability of a label.
 * 
 */
class Classifier {
  private:
  // DATA MEMBERS
  int numT; // Total number of posts in training set
  map<string, int> vocab; // Every unique word and their counts in training set
  map<string, int> tag_counts; // Number of times a tag appears in training
  map<string, double> tag_prob; // Stores the prior probabilities of each tag
  map<string, map<string, int>> word_counts; // Stores the counts for each word for each tag
  map<string, map<string, double>> word_prob; 
  // Stores word probabilites for processed values (tag, word, probability)

  public:

  // Constructor to initialize data values 
  Classifier() : numT(0) {} 

  // Destructor
  ~Classifier() = default;

  // EFFECT: clears variables if need be
  void initialize() {
    numT = 0;
    tag_counts.clear();
    word_counts.clear();
    tag_prob.clear();
    vocab.clear();
  }
    
  /**
   * REQUIRES: 
   *
   * BRIEF: takes input file stream and trains the classifier on the given data stream
   * 
   * PARAM: stream 
   */
  void train(const string& filename) {  
    csvstream csvin(filename); 
    map<string, string> row; // Expected to have "tag" and "content" fields

    while (csvin >> row) {
      numT++; //Increment total number of posts counted
      
      string tag = row["tag"];
      string content = row["content"]; 

      tag_counts[tag]++; // Increment the count for this tag

      set<string> unique_words_post = unique_words(content);

      // Update word counts for this tag
      for (const auto& word : unique_words_post) {
          word_counts[tag][word]++;  // Increment count for the word in this tag
          vocab[word]++;             // Increment global count for the word in vocab
      }
    }

    // log prior init
    for (const auto& tag_entry : tag_counts) {
      log_prior(tag_entry.first);
    }
  }

  /**
   * BRIEF: takes in a post and makes a prediction
   * 
   * PARAM: content of specified post
   * RETURN: string of the most probable label and log score
   */
  pair<string, double> predict(const string& content) {
    map <string, double> potentials; // to store all tags and potential probabilities (for guessing)
    set<string> unique_words_post = unique_words(content);


    // Iterate through each possible tag
    for (const auto& label : tag_prob){
      double total_log_prob = label.second; // Start with prior possibility

      // Iterates through each word in post
      for (const auto& word : unique_words_post) {
        total_log_prob += cal_word_prob(word, label.first); // log probability from each word
      }

      potentials[label.first] = total_log_prob;
    }
    
    // ---------------- Makes prediction ---------------------
    pair<std::string, double> maxPair = *potentials.begin(); // initialize maxPair
    for (const auto& pair : potentials) {
        if (pair.second > maxPair.second) {
            maxPair = pair;
        }
    }

    return maxPair; // Returns the most probable label and its log score
  }


  /**
  * REQUIRES: 
      1) that tag is a valid label in tag_counts and was seen during training 
      2) that all instances of a word in a label C exists and has a value of at least 1
      3) that numT is atleast 1 or greater
  * EFFECT: Returns the  LOGGED probability of a word 'w' in tag 'C' 
  * Case 1: If 'w' doesn't occur in 'C': ln P(w|C) = ln (num 'w'/num sets T)
  * Case 2: If 'w' doesn't occur ANYWHERE: ln P(w|C) = ln(1/num sets T)
  * Case 3: ln P(w|C) = ln (num sets 'C' & 'w' / num sets  'C')
  */
  double cal_word_prob(const string& word, const string& tag) {
    const auto& tag_word_counts = word_counts.at(tag); // tag_word_counts = C
    bool word_in_vocab = vocab.find(word) != vocab.end();

    // CASE 1: Checks if word is NOT in C but is in vocab
    if (tag_word_counts.find(word) == tag_word_counts.end() && word_in_vocab){
      return log(static_cast<double>(vocab[word])/ numT);
    }

    // CASE 2: Checks if word doesn't occur in vocab
    if (!word_in_vocab) { 
      return log(1.0 / numT); 
    } 

    // CASE 3: word does exist in label C
    return log(static_cast<double> (tag_word_counts.at(word)) / tag_counts[tag]);
  }

  /**
   * REQUIRES: 
      1) Tag is a valid label in tag_counts and was seen in training
   * EFFECT: Returns the logged probability of tag 'C' in number of training post 
   */
  void log_prior(const string& tag) {
    auto it = tag_counts.find(tag);
    if (it == tag_counts.end()) {
      cout << "Runtime Error: Label not found in training data";
    }
    
    tag_prob[tag] = log(static_cast<double> (tag_counts[tag]) / numT);
  }


  /**
   * EFFECTS: Return a set of unique whitespace delimited words
  */ 
  set<string> unique_words(const string& str) {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word) {
      words.insert(word);
    }
    return words;
  }
  
  // ACCESSOR FUNCTIONS
  /**
   * EFFECTS: Return the number of posts in training
  */
  int get_numT() const {
    return numT;
  } 

  int get_numV() const{
    return vocab.size();
  }

  const map<string, int>& get_tag_counts() const {
    return tag_counts;
  }

  const map<string, double>& get_tag_prob() const {
    return tag_prob;
  }

  const map<string, map<string, int>>& get_word_counts() const {
    return word_counts;
  }

  const map<string, map<string, double>>& get_word_prob() const{
    return word_prob;
  }
   
};



////////////////////////////////// MAIN FUNCTION ///////////////////////////

int main(int argc, char **argv) {
  cout.precision(3); // setting floating point precision
  bool trainOnly = argc != 3; //


  // ERROR CHECKING:

  // Checking correct amount of arguments passed.
  if (argc != 2 && trainOnly){ 
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }

  // Opening training file stream
  ifstream trainFile(argv[1]);
  if (!trainFile.is_open()) { // Error handling
    cout << "Error opening file: " << argv[1] << endl;
    return 1;
  }
  
  // Open test file stream
  ifstream testFile; //initialize testFile
  if(!trainOnly) {
    testFile.open(argv[2]); //opens testFile
    if (!testFile.is_open()) { // Error handling
      cout << "Error opening file: " << argv[2] << endl;
      return 1;
    } 
  } 

  // MAIN IMPLEMENTATION
  Classifier classifier;
  classifier.train(argv[1]);

  // Used to store the content 
  vector<vector<string>> bag_of_words;

  // TRAINING ONLY
  if(trainOnly){
    // SECTION 1
    int numT = classifier.get_numT();
    cout << "training data:" << endl;
    
    // Iterates through every given training label
    for (int i = 0; i < numT; i++) {
      string label, content;
      cout << "  label = " << label << ", content = " << content << endl;
    }
    cout << "trained on " << numT << " examples" << endl;
    cout << "vocabulary size = " << classifier.get_numV() << endl;
    cout << endl; // additional blank line as per spec

    // SECTION 2

    cout << "classes:" << endl;
    // Iterates through each label
    for (const auto& pair : classifier.get_tag_prob()) {
      const string& label = pair.first;
      int label_count = classifier.get_tag_counts().at(label);
      double log_value = pair.second;
      cout << "  " << label << ", " << label_count << ", log-prior = " << log_value << endl;
    }

    // SECTION 3
    
    cout << "classifier parameters:" << endl;
    // Iterates through each label
    const auto& word_counts = classifier.get_word_counts();
    for (const auto& tag : classifier.get_tag_counts()) {
      const string& label = tag.first;
      // Iterates through each word in label
      for (const auto& pair : tag.second) {
        const string& word = pair.first;
        int count = word_counts.at(label).at(word);
        double log_value = pair.second;
        cout << label << ":" << word << ", count = " << count << ", log-likelihood = " << log_value << endl;
      }
    }
    cout << endl; // Extra line as per spec    

  return 0;

  // TEST FILE
  } else {
    int numC = 0, numP = 0; // Total Correct Predictions, and Predictions

    cout << "trained on " << classifier.get_numT() << " examples" << endl << endl;

    csvstream csvin(argv[2]);
    map<string, string> row; 
    
    cout << "test data:" << endl;
    // Iterates though each post by row in test data
    while (csvin >> row) {
      string correct_tag = row["tag"];
      string content = row["content"];

      const auto& prediction = classifier.predict(content);
      cout << "  correct = " << correct_tag << prediction.first << ", log-probability score = " << prediction.second << endl;
      if (correct_tag == prediction.first) { ++numC;} // Tracks the number correct predictions
      cout << "  content = " << content << endl << endl; // extra line as per spec
      ++numP;
    }

    cout << "performance: " << numC << " / " << numP << " posts predicted correctly" << endl;
  }

  return 0;
}