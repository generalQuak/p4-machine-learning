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

class Classifier {
  private:
  // DATA MEMBERS
  int numT; 
  vector<string> content; 
  map<string, int> vocab; 
  map<string, int> tag_counts; 
  map<string, double> tag_prob; 
  map<string, map<string, int>> word_counts;
  map<string, map<string, double>> word_prob;

  public:

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
    content.clear();
  }
    
  void train(const string& filename) {  
    csvstream csvin(filename); 
    map<string, string> row;

    while (csvin >> row) {
      numT++; //Increment total number of posts counted
      
      string tag = row["tag"];
      string content = row["content"]; 

      tag_counts[tag]++; // Increment the count for this tag

      set<string> unique_words_post = unique_words(content);

      // Update word counts for this tag
      for (const auto& word : unique_words_post) {
          word_counts[tag][word]++;
          vocab[word]++;
      }
    }

    // log prior init
    for (const auto& tag_entry : tag_counts) {
      log_prior(tag_entry.first);
    }
  }

  pair<string, double> predict(const string& content) {
    map <string, double> potentials; 
    set<string> unique_words_post = unique_words(content);


    // Iterate through each possible tag
    for (const auto& label : tag_prob){
      double total_log_prob = label.second; 

      // Iterates through each word in post
      for (const auto& word : unique_words_post) {
        total_log_prob += cal_word_prob(word, label.first); 
      }

      potentials[label.first] = total_log_prob;
    }
    
    // ---------------- Makes prediction ---------------------
    pair<std::string, double> maxPair = *potentials.begin(); 
    for (const auto& pair : potentials) {
        if (pair.second > maxPair.second) {
            maxPair = pair;
        }
    }

    return maxPair; 
  }

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

  void log_prior(const string& tag) {
    auto it = tag_counts.find(tag);
    if (it == tag_counts.end()) {
      cout << "Runtime Error: Label not found in training data";
    }
    
    tag_prob[tag] = log(static_cast<double> (tag_counts[tag]) / numT);
  }

  set<string> unique_words(const string& str) {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word) {
      words.insert(word);
    }
    return words;
  }
  
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
    bool trainOnly = argc != 3;

    if (!validate_arguments(argc, argv, trainOnly)) {
        return 1;
    }

    Classifier classifier;
    classifier.train(argv[1]);

    if (trainOnly) {
        display_training_data(argv[1]);
        display_classifier_parameters(classifier);
    } else {
        display_test_data(argv[2], classifier);
    }

    return 0;
}

// Validates arguments and opens files
bool validate_arguments(int argc, char **argv, bool trainOnly) {
    if (argc != 2 && trainOnly) {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return false;
    }

    ifstream trainFile(argv[1]);
    if (!trainFile.is_open()) {
        cout << "Error opening file: " << argv[1] << endl;
        return false;
    }
    
    if (!trainOnly) {
        ifstream testFile(argv[2]);
        if (!testFile.is_open()) {
            cout << "Error opening file: " << argv[2] << endl;
            return false;
        }
    }

    return true;
}

// Displays training data
void display_training_data(const string& train_file) {
    cout << "training data:" << endl;
    csvstream csvin(train_file);
    map<string, string> row;
    while (csvin >> row) {
        string label = row["tag"];
        string content = row["content"];
        cout << "  label = " << label << ", content = " << content << endl;
    }
}

// Displays classifier parameters
void display_classifier_parameters(const Classifier& classifier) {
    cout << "classifier parameters:" << endl;
    const auto& word_counts = classifier.get_word_counts();
    for (const auto& tag : word_counts) {
        const string& label = tag.first;
        cout << "  " << label << ", " << classifier.get_label_count(label) << " examples, log-prior = "
             << classifier.get_log_prior(label) << endl;

        for (const auto& word_entry : tag.second) {
            const string& word = word_entry.first;
            cout << "  " << label << ":" << word << ", count = " << word_counts.at(label).at(word)
                 << ", log-likelihood = " << classifier.cal_word_prob(word, label) << endl;
        }
    }
    cout << endl;
}

// Displays test data results
void display_test_data(const string& test_file, Classifier& classifier) {
    int num_correct = 0, num_total = 0;
    cout << "test data:" << endl;

    csvstream csvin(test_file);
    map<string, string> row;
    while (csvin >> row) {
        string correct_tag = row["tag"];
        string content = row["content"];
        const auto& prediction = classifier.predict(content);

        cout << "  correct = " << correct_tag << ", predicted = " << prediction.first
             << ", log-probability score = " << prediction.second << endl;
        cout << "  content = " << content << endl << endl;

        if (correct_tag == prediction.first) {
            ++num_correct;
        }
        ++num_total;
    }

    cout << "performance: " << num_correct << " / " << num_total << " posts predicted correctly" << endl;
}