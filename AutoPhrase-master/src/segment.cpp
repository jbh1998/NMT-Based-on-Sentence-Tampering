#include "utils/parameters.h"
#include "utils/commandline_flags.h"
#include "utils/utils.h"
#include "frequent_pattern_mining/frequent_pattern_mining.h"
#include "data/documents.h"
#include "classification/feature_extraction.h"
#include "classification/label_generation.h"
#include "classification/predict_quality.h"
#include "model_training/segmentation.h"
#include "data/dump.h"

using FrequentPatternMining::Pattern;
using FrequentPatternMining::patterns;

vector<double> f;
vector<int> pre;

int highlights = 0, sentences = 0;

void process(const vector<TOTAL_TOKENS_TYPE>& tokens, const vector<POS_ID_TYPE>& tags, Segmentation& segmenter, FILE* out, FILE* myout)
{
    ++ sentences;
    if (ENABLE_POS_TAGGING) {
        segmenter.viterbi_for_testing(tokens, tags, f, pre, SEGMENT_MULTI_WORD_QUALITY_THRESHOLD, SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
    } else {
        segmenter.viterbi_for_testing(tokens, f, pre, SEGMENT_MULTI_WORD_QUALITY_THRESHOLD, SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
    }

    int i = (int)tokens.size();
    assert(f[i] > -1e80);
    vector<string> ret;
    vector<int> my_ret;
    while (i > 0) {
        int j = pre[i];
        size_t u = 0;
        bool quality = true;
        for (int k = j; k < i; ++ k) {
            if (!trie[u].children.count(tokens[k])) {
                quality = false;
                break;
            }
            u = trie[u].children[tokens[k]];
        }
        quality &= trie[u].id == patterns.size() && ( // These phrases are in the wiki_quality.txt, their quality scores are treated as 1.
                        i - j > 1 && 1 >= SEGMENT_MULTI_WORD_QUALITY_THRESHOLD ||
                        i - j == 1 && 1 >= SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD) || 
                   trie[u].id < patterns.size() && trie[u].id >= 0 && (
                        patterns[trie[u].id].size() > 1 && patterns[trie[u].id].quality >= SEGMENT_MULTI_WORD_QUALITY_THRESHOLD ||
                        patterns[trie[u].id].size() == 1 && patterns[trie[u].id].quality >= SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD
                   );
        if (quality) {
            ret.push_back("</phrase>");
            ++ highlights;
        }
        my_ret.push_back(1);
        for( int k = i-2; k >= j ;--k)
            my_ret.push_back(0);
        for (int k = i - 1; k >= j; -- k) {
            ostringstream sout;
            sout << tokens[k];
            ret.push_back(sout.str());
        }
        if (quality) {
            ret.push_back("<phrase>");
        }

        i = j;
    }

    reverse(ret.begin(), ret.end());
    reverse(my_ret.begin(), my_ret.end());
    for (int i = 0; i < ret.size(); ++ i) {
        fprintf(out, "%s%c", ret[i].c_str(), i + 1 == ret.size() ? '\n' : ' ');
    }
    myAssert((int)my_ret.size() == (int)tokens.size(), "size not equal");
    for (int i = 0; i < my_ret.size(); ++ i)
        fprintf(myout," %d",my_ret[i]);
}

inline bool byQuality(const Pattern& a, const Pattern& b)
{
    return a.quality > b.quality + EPS || fabs(a.quality - b.quality) < EPS && a.currentFreq > b.currentFreq;
}

int main(int argc, char* argv[])
{
    parseCommandFlags(argc, argv);

    sscanf(argv[1], "%d", &NTHREADS);
    omp_set_num_threads(NTHREADS);

    Dump::loadSegmentationModel(SEGMENTATION_MODEL);

    sort(patterns.begin(), patterns.end(), byQuality);

    constructTrie(false); // update the current frequent enough patterns

    Segmentation* segmenter;
    if (ENABLE_POS_TAGGING) {
        segmenter = new Segmentation(ENABLE_POS_TAGGING);
        Segmentation::getDisconnect();
        Segmentation::logPosTags();
    } else {
        segmenter = new Segmentation(Segmentation::penalty);
    }

    char currentTag[100];

    FILE* in = tryOpen(TEXT_TO_SEG_FILE, "r");
    FILE* posIn = NULL;
    if (ENABLE_POS_TAGGING) {
        posIn = tryOpen(TEXT_TO_SEG_POS_TAGS_FILE, "r");
    }

    FILE* out = tryOpen("tmp/tokenized_segmented_sentences.txt", "w");
    FILE* myout = tryOpen("tmp/tokenized_segmented_out.txt", "w");

    while (getLine(in)) {
        stringstream sin(line);
        vector<TOTAL_TOKENS_TYPE> tokens;
        vector<POS_ID_TYPE> tags;

        string lastPunc = "";
        for (string temp; sin >> temp;) {
            // get pos tag
            POS_ID_TYPE posTagId = -1;
            if (ENABLE_POS_TAGGING) {
                myAssert(fscanf(posIn, "%s", currentTag) == 1, "POS file doesn't have enough POS tags");
                if (!Documents::posTag2id.count(currentTag)) {
                    posTagId = -1; // unknown tag
                } else {
                    posTagId = Documents::posTag2id[currentTag];
                }
            }

            // get token
            bool flag = true;
            TOKEN_ID_TYPE token = 0;
            for (size_t i = 0; i < temp.size() && flag; ++ i) {
                flag &= isdigit(temp[i]) || i == 0 && temp.size() > 1 && temp[0] == '-';
            }
            stringstream sin(temp);
            sin >> token;

            if (!flag) {
                string punc = temp;
                if( Documents::separatePunc.count(punc) <= 0)
                    printf("%s\n",temp.c_str());
                process(tokens, tags, *segmenter, out, myout);
                tokens.clear();
                tags.clear();
                fprintf(myout, " 1");
            } else {
                tokens.push_back(token);
                if (ENABLE_POS_TAGGING) {
                    tags.push_back(posTagId);
                }
            }
        }
        if (tokens.size() > 0) {
            process(tokens, tags, *segmenter, out, myout);
        }
        fprintf(myout,"\n");
    }
    fclose(in);
    if (ENABLE_POS_TAGGING) {
        fclose(posIn);
    }
    fclose(out);
    fclose(myout);

    cerr << "Phrasal segmentation finished." << endl;
    cerr << "   # of total highlighted quality phrases = " << highlights << endl;
    cerr << "   # of total processed sentences = " << sentences << endl;
    cerr << "   avg highlights per sentence = " << (double)highlights / sentences << endl;

    return 0;
}
