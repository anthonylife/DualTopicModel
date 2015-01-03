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
// Running corresponding model (GSDMM for short text)            //
///////////////////////////////////////////////////////////////////

#include "model.hpp"

using namespace std;


void print_usage_and_exit() {
    // print usage information
    printf("*****Gibbs Sampling for Dirichlet Multinomial Mixture Model****\n");
    printf("Authors: anthonylife, xxx.gmail.com, Computer Science Department, XXX University.\n");
    printf("usage:\n");
    printf("      ./main [options]\n");
    printf("      --help:           print help information\n");

    printf("\n");
    printf("      -d:      data set choice\n");
    printf("      -r:      whether to restart train or using existing model\n");
    printf("      -tm:     which method to train the model\n");

    printf("******************************************************************************\n");
    exit(0);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


int main(int argc, char **argv) {
    //***Method variables needed to be specified by User (Default values)****//
    string trdata_path      =   "../../data/yelp_train.dat";
    string vadata_path      =   "../../data/yelp_vali.dat";
    string tedata_path      =   "../../data/yelp_test.dat";
    string model_path       =   "model.out";
    bool   restart_tag      =   false;
    string submission_path  =   "../../results/dsactm_result1.dat";
    ///////////////////////////////////////////////////////////////////////////// 
    
    int i;
    int data_num=-1;
    char *b=NULL;
    if (argc == 1) {
        printf("GSDMM v 0.1a\n");
        print_usage_and_exit();
        return 0;
    }
    if ((i = ArgPos((char *)"--help", argc, argv)) > 0) print_usage_and_exit();
    if ((i = ArgPos((char *)"-d", argc, argv)) > 0) data_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-r", argc, argv)) > 0) b = argv[i + 1];
    //if ((i = ArgPos((char *)"-niter", argc, argv)) > 0) niters = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-minibatch", argc, argv)) > 0)
    //    minibatch = atoi(argv[i + 1]);
    
    if (data_num!=0 && data_num!=1 && data_num!=2 && data_num!=3 &&
            data_num!=4 && data_num!=5 && data_num!=6 && data_num!=7) {
        printf("Invalid choice of dataset!\n");
        exit(1);
    } else if (data_num==0) {
        trdata_path = "../../data/yelp_train.dat";
        vadata_path = "../../data/yelp_vali.dat";
        tedata_path = "../../data/yelp_test.dat";
        submission_path = "../../results/dsactm_result1.dat";
    } else if (data_num==1) {
        trdata_path = "../../data/amazonfood_train.dat";
        vadata_path = "../../data/amazonfood_vali.dat";
        tedata_path = "../../data/amazonfood_test.dat";
        submission_path = "../../results/dsactm_result2.dat";
    } else if (data_num==2) {
        trdata_path = "../../data/amazonmovie_train.dat";
        vadata_path = "../../data/amazonmovie_vali.dat";
        tedata_path = "../../data/amazonmovie_test.dat";
        submission_path = "../../results/dsactm_result3.dat";
    } else if (data_num==3){
        trdata_path = "../../data/arts_train.dat";
        vadata_path = "../../data/arts_vali.dat";
        tedata_path = "../../data/arts_test.dat";
        submission_path = "../../results/dsactm_result4.dat";
    } else if (data_num==4){
        trdata_path = "../../data/yelp_tip_train.dat";
        vadata_path = "../../data/yelp_tip_vali.dat";
        tedata_path = "../../data/yelp_tip_test.dat";
        submission_path = "../../results/dsactm_result5.dat";
    } else if (data_num==5){
        trdata_path = "../../data/cellartracker_train.dat";
        vadata_path = "../../data/cellartracker_vali.dat";
        tedata_path = "../../data/cellartracker_test.dat";
        submission_path = "../../results/dsactm_result6.dat";
    } else if (data_num==6){
        trdata_path = "../../data/video_train.dat";
        vadata_path = "../../data/video_vali.dat";
        tedata_path = "../../data/video_test.dat";
        submission_path = "../../results/dsactm_result7.dat";
    } else if (data_num==7){
        trdata_path = "../../data/sports_train.dat";
        vadata_path = "../../data/sports_vali.dat";
        tedata_path = "../../data/sports_test.dat";
        submission_path = "../../results/dsactm_result8.dat";
    } else {
        printf("Invalid choice of dataset\n");
        exit(1);
    }
    
    if (strcmp(b, (char *)"True") == 0)
        restart_tag = true;
    else if (strcmp(b, (char *)"False") == 0)
        restart_tag = false;
    else {
        printf("Invalid input of para -r\n");
        exit(1);
    }
   
    printf("GSDMM v 0.1a\n");
    timeval start_t, end_t;
    utils::tic(start_t);
    GSDMM * gsdmm = new GSDMM((char *)trdata_path.c_str(),
                                (char *)vadata_path.c_str(),
                                (char *)tedata_path.c_str(),
                                (char *)model_path.c_str(),
                                restart_tag);
    if (restart_tag)
        gsdmm->train();
    gsdmm->inference();
    utils::toc(start_t, end_t);

    return 0;
}
