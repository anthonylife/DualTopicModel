#pragma once
//Copyright [2014] [Wei Zhang]

//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

///////////////////////////////////////////////////////////////////
// Date: 2014/12/30                                              //
// Model Implementation (GSDMM).                                 //
///////////////////////////////////////////////////////////////////

#include "../utils.hpp"
#include "corpus.hpp"

using namespace std;
using namespace __gnu_cxx;

#define MINVAL 1e-50

class GSDMM{
    //******Model parameter needed to be specified by User*******//
    /// Method control variable                                  //
    const static int ngibbs_train = 100;                        //
    const static int ngibbs_inf = 100;                           //
    const static int K = 40;                                     //
    const static int max_words = 8000;                          //
    constexpr static double alpha = 0.1;                         //
    constexpr static double beta = 0.05;                          //
    ///////////////////////////////////////////////////////////////

    int * N_dz;
    int ** N_w_z;
    int * N_sumwz;
    int * newN_dz;
    int ** newN_w_z;
    int * newN_sumwz;
    int * D_z;
    int * newD_z;
    double * theta_dk;
    double * newtheta_dk;
    double ** beta_vk;
    double ** newbeta_vk;

    Corpus* corp;
    vector<Vote*> train_votes;
    vector<Vote*> vali_votes;
    vector<Vote*> test_votes;
    int n_users;
    int n_items;
    int n_words;

    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;

public:
    GSDMM(char* trdata_path, char* vadata_path, char* tedata_path,
            char* model_path, bool restart_tag) {
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->restart_tag = restart_tag;

        printf("Loading data.\n");
        corp = new Corpus(trdata_path, vadata_path, tedata_path, max_words);
        n_users = corp->n_users;
        n_items = corp->n_items;
        n_words = corp->n_words;
        printf("n_words: %d\n", n_words);
        for (vector<Vote*>::iterator it = corp->TR_V->begin(); 
                it != corp->TR_V->end(); it++) {
            if ((int)(*it)->words.size() > 0) 
                train_votes.push_back(*it);
        }
        for (vector<Vote*>::iterator it = corp->TE_V->begin();
                it != corp->TE_V->end(); it++) {
            if ((int)(*it)->words.size() > 0) 
                test_votes.push_back(*it);
        }
        for (vector<Vote*>::iterator it = corp->VA_V->begin();
                it != corp->VA_V->end(); it++) {
            if ((int)(*it)->words.size() > 0) 
                vali_votes.push_back(*it);
        }
        
        if (restart_tag == true) {
            printf("Para initialization from restart.\n");
            modelParaInit();
        } else {
            printf("Para loading from trained model.\n");
            loadModelPara();
        }
        printf("Finishing all initialization.\n");
    }
    
    ~GSDMM() {
        delete corp;
        delete[] N_dz;
        delete[] N_sumwz;
        for (int w=0; w<n_words; w++)
            delete[] N_w_z[w];
        delete[] N_w_z;
        delete[] newN_dz;
        delete[] newN_sumwz;
        for (int w=0; w<n_words; w++)
            delete[] newN_w_z[w];
        delete[] newN_w_z;

        train_votes.clear();
        vector<Vote*>(train_votes).swap(train_votes);
        vali_votes.clear();
        vector<Vote*>(vali_votes).swap(vali_votes);
        test_votes.clear();
        vector<Vote*>(test_votes).swap(test_votes);
    }

    void modelParaInit() {
        D_z = new int[train_votes.size()];
        memset(D_z, 0, sizeof(int)*train_votes.size());
        N_dz = new int[K];
        memset(N_dz, 0, sizeof(int)*K);
        N_sumwz = new int[K];
        memset(N_sumwz, 0, sizeof(int)*K);
        N_w_z = utils::alloc_matrix(n_words, K, 0);
        utils::set_matrix(N_w_z, n_words, K, 0);
        theta_dk = new double[K];
        beta_vk = utils::alloc_matrix(n_words, K, (double)0.0);

        newD_z = new int[test_votes.size()];
        memset(newD_z, 0, sizeof(int)*test_votes.size());
        newN_dz = new int[K];
        memset(newN_dz, 0, sizeof(int)*K);
        newN_sumwz = new int[K];
        memset(newN_sumwz, 0, sizeof(int)*K);
        newN_w_z = utils::alloc_matrix(n_words, K, (int)0);
        utils::set_matrix(newN_w_z, n_words, K, 0);
        newtheta_dk = new double[K];
        newbeta_vk = utils::alloc_matrix(n_words, K, (double)0.0);
    }

    void train() {
        samplingInit(train_votes, D_z, N_dz, N_sumwz, N_w_z);
        gibbsSamplingTrain();
        saveModelPara();
    }

    void samplingInit(vector<Vote*> votes,int * D_z, int * N_dz,
            int * N_sumwz,int ** N_w_z) {
        int ind = -1;
        for (vector<Vote*>::iterator it=votes.begin();
                it!=votes.end(); it++) {
            ind++;
            int word_num = (*it)->words.size();
            if (word_num == 0)
                continue;
            int topic = (int)(((double)random()/RAND_MAX)*K);
            D_z[ind] = topic;
            N_dz[topic]++;
            N_sumwz[topic] += word_num;
            for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                    it_w!=(*it)->wordcnt.end(); it_w++) 
                N_w_z[it_w->wid][topic] += it_w->wcnt;
        }
    }

    void gibbsSamplingTrain() {
        int topic, review_nwords, stopic, ind, ind1;
        double sval, train_perp, Vbeta=n_words*beta, Kalpha=K*alpha;
        long double * topic_prob = new long double[K];
        long double * cum_prob = new long double[K];

        timeval start_t, end_t;
        utils::tic(start_t);
        printf("haha\n");
        cout.setf(ios::scientific);
        for (int iter=0; iter<ngibbs_train; iter++) {
            ind = -1;
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                ind++;
                review_nwords = (int)(*it)->words.size();
                if (review_nwords==0)
                    continue;
                topic = D_z[ind];
                N_dz[topic]--;
                N_sumwz[topic] -= review_nwords;
                for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                        it_w!=(*it)->wordcnt.end(); it_w++)
                    N_w_z[it_w->wid][topic] -= it_w->wcnt;
                for (int k=0; k<K; k++) {
                    topic_prob[k] = (N_dz[k]+alpha)/(train_votes.size()-1+Kalpha);
                    ind1 = 1;
                    for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                            it_w!=(*it)->wordcnt.end(); it_w++) {
                        for (int j=1; j<=it_w->wcnt; j++) {
                            topic_prob[k] *= (N_w_z[it_w->wid][k]+beta+j-1)/(N_sumwz[k]+Vbeta+ind1-1);
                            ind1++;
                        }
                    }
                    if (k==0) {
                        cum_prob[k] = topic_prob[k];
                        //cout << "0-->" << cum_prob[k] << ":" << topic_prob[k] << endl;
                        //utils::pause();
                    } else {
                        cum_prob[k] = cum_prob[k-1] + topic_prob[k];
                        //cout << k << "-->" << cum_prob[k] << ":" << topic_prob[k] << endl;
                        //utils::pause();
                    }
                }
                sval = ((double)random()/RAND_MAX)*cum_prob[K-1];
                for (stopic=0; stopic<K; stopic++) {
                    if (sval < cum_prob[stopic])
                        break;
                } 
                //printf("chooce topic: %d\n", stopic);
                //utils::pause();
                D_z[ind] = stopic;
                N_dz[stopic]++;
                N_sumwz[stopic] += review_nwords;
                for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                        it_w!=(*it)->wordcnt.end(); it_w++)
                    N_w_z[it_w->wid][stopic] += it_w->wcnt;
            }
            computeTheta();
            computeBeta();
            evalPerplexity(train_perp, &train_votes, D_z);
            printf("Current iteration: %d, train perplexity: %lf", iter, train_perp);
            utils::toc(start_t, end_t, true);
            utils::tic(start_t);
        }  
        delete[] topic_prob;
        delete[] cum_prob;
    }

    void inference() {
        samplingInit(test_votes, newD_z, newN_dz, newN_sumwz, newN_w_z);
        gibbsSamplingInf();
    }
   
    void gibbsSamplingInf() {
        int topic, review_nwords, stopic, ind, ind1;
        double test_perp;
        double sval, Vbeta=n_words*beta, Kalpha=K*alpha;
        long double * topic_prob = new long double[K];
        long double * cum_prob = new long double[K];
        
        timeval start_t, end_t;
        utils::tic(start_t);
        for (int iter=0; iter<ngibbs_inf; iter++) {
            ind = -1;
            for (vector<Vote*>::iterator it=test_votes.begin();
                    it!=test_votes.end(); it++) {
                ind++;
                review_nwords = (int)(*it)->words.size();
                if (review_nwords==0)
                    continue;
                topic = newD_z[ind];
                newN_dz[topic]--;
                newN_sumwz[topic] -= review_nwords;
                for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                        it_w!=(*it)->wordcnt.end(); it_w++)
                    newN_w_z[it_w->wid][topic] -= it_w->wcnt;
                for (int k=0; k<K; k++) {
                    topic_prob[k] = (N_dz[k]+newN_sumwz[k]+alpha)/(train_votes.size()+test_votes.size()-1+Kalpha);
                    ind1 = 1;
                    for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                            it_w!=(*it)->wordcnt.end(); it_w++) {
                        for (int j=1; j<=it_w->wcnt; j++) {
                            topic_prob[k] *= (N_w_z[it_w->wid][k]+newN_w_z[it_w->wid][k]+beta+j-1)/(N_sumwz[k]+newN_sumwz[k]+Vbeta+ind1-1);
                            ind1++;
                        }
                    }
                    if (k==0)
                        cum_prob[k] = topic_prob[k];
                    else
                        cum_prob[k] = cum_prob[k-1] + topic_prob[k];
                }
                sval = ((double)random()/RAND_MAX)*cum_prob[K-1];
                for (stopic=0; stopic<K; stopic++) {
                    if (sval < cum_prob[stopic])
                        break;
                } 
                newD_z[ind] = stopic;
                newN_dz[stopic]++;
                newN_sumwz[stopic] += review_nwords;
                for (vector<Cword>::iterator it_w=(*it)->wordcnt.begin();
                        it_w!=(*it)->wordcnt.end(); it_w++)
                    newN_w_z[it_w->wid][stopic] += it_w->wcnt;
            }
            computeNewTheta();
            computeNewBeta();
            evalPerplexity(test_perp, &test_votes, newD_z);
            printf("Current iteration: %d, test perplexity: %lf", iter, test_perp);
            utils::toc(start_t, end_t, true);
            utils::tic(start_t);
        }  
        delete[] topic_prob;
        delete[] cum_prob;
    }

    void computeTheta() {
        double Kalpha = K*alpha;
        for (int k=0; k<K; k++)
            theta_dk[k] = (N_dz[k]+alpha)/(train_votes.size()+Kalpha);
    }

    void computeBeta() {
        double Vbeta=n_words*beta;
        for (int k=0; k<K; k++)
            for (int w=0; w<n_words; w++)
                beta_vk[w][k] = (N_w_z[w][k]+beta)/(N_sumwz[k]+Vbeta);
    }
    
    void computeNewTheta() {
        double Kalpha = K*alpha;
        for (int k=0; k<K; k++)
            theta_dk[k] = (N_dz[k]+newN_dz[k]+alpha)/(train_votes.size()+test_votes.size()+Kalpha);
    }
    
    void computeNewBeta() {
        double Vbeta=n_words*beta;
        for (int k=0; k<K; k++)
            for (int w=0; w<n_words; w++)
                beta_vk[w][k] = (N_w_z[w][k]+newN_w_z[w][k]+beta)/(N_sumwz[k]+newN_sumwz[k]+Vbeta);
    }

    /*void evalPerplexity(double& perp, int label) {
        int word_cnt = 0;
        long double word_log_prob, cache_val;
        double * doc_topic_prob = new double[K];
        
        perp = 0;
        if (label == 0) {
            for (vector<Vote*>::iterator it = train_votes.begin();
                    it != train_votes.end(); it++) {
                word_log_prob = 0.0;
                word_cnt += (*it)->words.size();
                if ((*it)->words.size() == 0)
                    continue;
                for (int k=0; k<K; k++) {
                    cache_val = theta_dk[k];
                    for (vector<int>::iterator it1 = (*it)->words.begin();
                            it1!=(*it)->words.end(); it1++) { 
                        cache_val *= beta_vk[*it1][k];
                    }
                    word_log_prob += cache_val;
                }
                perp += log(word_log_prob);
            }
            perp = exp(-perp/word_cnt);
        } else if (label == 1) {
            for (vector<Vote*>::iterator it = test_votes.begin();
                    it != test_votes.end(); it++) {
                word_log_prob = 0.0;
                word_cnt += (*it)->words.size();
                if ((*it)->words.size() == 0)
                    continue;
                for (int k=0; k<K; k++) {
                    cache_val = theta_dk[k];
                    for (vector<int>::iterator it1 = (*it)->words.begin();
                            it1!=(*it)->words.end(); it1++) { 
                        cache_val *= beta_vk[*it1][k];
                    }
                    word_log_prob += cache_val;
                }
                perp += log(word_log_prob);
            }
            perp = exp(-perp/word_cnt);
        } else {
            printf("Invalid choice of test data set.\n");
            exit(1);
        }
    }*/
    
    void evalPerplexity(double& perp, vector<Vote*> * votes, int * Z) {
        int word_cnt = 0, topic, ind;
        long double word_log_prob, cache_val;
        double * doc_topic_prob = new double[K];
        
        perp = 0;
        cache_val = 0;
        ind = -1;
        //ofstream fd("record.txt");
        for (vector<Vote*>::iterator it = votes->begin();
                it != votes->end(); it++) {
            ind++;
            word_cnt += (*it)->words.size();
            if ((*it)->words.size() == 0)
                continue;
            topic = Z[ind];
            for (vector<int>::iterator it1 = (*it)->words.begin();
                        it1!=(*it)->words.end(); it1++) { 
                word_log_prob = beta_vk[*it1][topic];
                if (std::isnan(word_log_prob || std::isinf(word_log_prob) || word_log_prob==0)) {
                    printf("word_log_prob: %lf\n", word_log_prob);
                    utils::pause();
                }
                if (word_log_prob < MINVAL) {
                    word_log_prob = MINVAL;
                }
                cache_val += log(word_log_prob);
                if (std::isnan(cache_val || std::isinf(cache_val))) {
                    printf("word_log_prob: %lf\n", word_log_prob);
                    utils::pause();
                }
                //fd << "cache_val: " << cache_val << "ind: " << ind << endl;
            }
        }
        //cout << "cache_val: " << cache_val << endl;
        //utils::pause();
        perp = exp(-cache_val/word_cnt);
    }
    
    void saveModelPara() {
        FILE* f = utils::fopen_(model_path, "w");
        fwrite(D_z, sizeof(int), train_votes.size(), f);
        fwrite(N_dz, sizeof(int), K, f);
        fwrite(N_sumwz, sizeof(int), K, f);
        for (int w=0; w<n_words; w++) {
            fwrite(N_w_z[w], sizeof(int), K, f);
        }
        fclose(f);
    }

    void loadModelPara() {
        D_z = new int[train_votes.size()];
        memset(D_z, 0, sizeof(int)*train_votes.size());
        N_dz = new int[K];
        memset(N_dz, 0, sizeof(int)*K);
        N_sumwz = new int[K];
        memset(N_sumwz, 0, sizeof(int)*K);
        N_w_z = utils::alloc_matrix(n_words, K, (int)0);
        utils::set_matrix(N_w_z, n_words, K, 0);
        theta_dk = new double[K];
        beta_vk = utils::alloc_matrix(n_words, K, (double)0.0);

        newD_z = new int[test_votes.size()];
        memset(newD_z, 0, sizeof(int)*test_votes.size());
        newN_dz = new int[K];
        memset(newN_dz, 0, sizeof(int)*K);
        newN_sumwz = new int[K];
        memset(newN_sumwz, 0, sizeof(int)*K);
        newN_w_z = utils::alloc_matrix(n_words, K, (int)0);
        utils::set_matrix(newN_w_z, n_words, K, 0);
        newtheta_dk = new double[K];
        newbeta_vk = utils::alloc_matrix(n_words, K, (double)0.0);
        
        // total number of paramters to be learned
        FILE* f = utils::fopen_(model_path, "r");
        fread(D_z, sizeof(int), train_votes.size(), f);
        fread(N_dz, sizeof(int), K, f);
        fread(N_sumwz, sizeof(int), K, f);
        for (int w=0; w<n_words; w++) {
            fread(N_w_z[w], sizeof(int), K, f);
        }
        fclose(f);
    }

};
