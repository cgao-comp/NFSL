import sys

import networkx as nx
import numpy as np
from tqdm import tqdm
import processing16
import scipy

from utils import *
from data import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from processing16 import DataReader


def create_graph1(args):
    if args.graph_type == 'Twitter':
        
        with open('twitter16.pkl', 'rb') as f:
            twitter16 = pickle.load(f)

        def convert_date_string(date_string):
            
            pattern = r"(\d{4})年(\d{1,2})月"
            match = re.match(pattern, date_string)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                
                result = year + month * 0.01
                if type(result) != float:
                    print(date_string)
                return result

        def processing_DAG(one_DAG, dropout_rate=False):
            if dropout_rate is not False:
                nodes_to_remove = []
                root = [node for node, in_degree in one_DAG.in_degree() if in_degree == 0][0]
                for neighbor in one_DAG.neighbors(root):
                    if one_DAG.out_degree(neighbor) == 0:
                        if random.random() < dropout_rate:
                            nodes_to_remove.append(neighbor)
                for node in nodes_to_remove:
                    one_DAG.remove_node(node)
            tweets_l = []
            reg_l = []
            followings_l = []
            fans_l = []
            ren_l = []
            ratio_l = []
            all_user = 0
            for one_user in one_DAG.nodes:
                all_user += 1
                att_list = one_DAG.nodes.get(one_user)['att']
                if float(att_list[0]) != -1.111111111 and float(att_list[0]) != -999:
                    tweets_l.append(float(att_list[0]))
                    convert_reg_data = (convert_date_string(att_list[1]))
                    if convert_reg_data is not None:
                        reg_l.append(convert_reg_data)
                        
                    else:
                        reg_l.append(2006)  
                    followings_l.append(float(att_list[2]))
                    fans_l.append(float(att_list[3]))
                    ren_l.append(1 if att_list[4] == 'True' else 0)

                    if float(att_list[2]) > 0 and float(att_list[3]) > 0:
                        ratio_l.append(float(att_list[3]) / float(att_list[2]))
                    else:
                        ratio_l.append(0)


            
            return sum(tweets_l) / all_user, sum(reg_l) / all_user, sum(followings_l) / all_user, sum(
                fans_l) / all_user, sum(ren_l) / all_user, sum(ratio_l) / all_user

        rumor_y = []
        common_y = []

        tweets_L_R = []
        reg_L_R = []
        followings_L_R = []
        fans_L_R = []
        ren_L_R = []
        ratio_L_R = []

        tweets_L_C = []
        reg_L_C = []
        followings_L_C = []
        fans_L_C = []
        ren_L_C = []
        ratio_L_C = []

        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'unverified':
                continue

            if y == 'true':  
                rumor_y.append(y)
                tweets, reg_date, followings, fans, is_varify, ratio = processing_DAG(one_DAG,
                                                                                      dropout_rate=args.dropout_rate)
                tweets_L_R.append(tweets)
                reg_L_R.append(reg_date)
                followings_L_R.append(followings)
                fans_L_R.append(fans)
                ren_L_R.append(is_varify)
                ratio_L_R.append(ratio)
            else:  
                common_y.append(y)
                tweets, reg_date, followings, fans, is_varify, ratio = processing_DAG(one_DAG,
                                                                                      dropout_rate=args.dropout_rate)
                tweets_L_C.append(tweets)
                reg_L_C.append(reg_date)
                followings_L_C.append(followings)
                fans_L_C.append(fans)
                ren_L_C.append(is_varify)
                ratio_L_C.append(ratio)

        combined_array_R = np.column_stack((np.array(tweets_L_R), np.array(reg_L_R), np.array(followings_L_R),
                                            np.array(fans_L_R), np.array(ren_L_R), np.array(ratio_L_R)))
        combined_array_C = np.column_stack((np.array(tweets_L_C), np.array(reg_L_C), np.array(followings_L_C),
                                            np.array(fans_L_C), np.array(ren_L_C), np.array(ratio_L_C)))
        combined_array = np.concatenate((combined_array_R, combined_array_C), axis=0)
        y_array = np.concatenate((np.array(rumor_y), np.array(common_y)))
        scaler = MinMaxScaler()
        standardized_data = scaler.fit_transform(combined_array)
        mapped_array = np.array([1 if y == "true" else 0 for y in y_array])

        k = 6  
        selector = SelectKBest(chi2, k=k)
        selector.fit(standardized_data, y_array)

        
        feature_scores = selector.scores_
        feature_ranks = np.argsort(-feature_scores)
        print("Feature scores:", feature_scores)
        print("Feature importance ranking (from most to least important):", feature_ranks)

        
        

        feature_scores_norm = feature_scores / sum(feature_scores)
        print(feature_scores_norm)
        

        def importance_comp(one_DAG,
                norm_info_ordering=[0.7486231, 0.11962602, 0.05546273, 0.04472802, 0.02356193, 0.00799819]):
                
                
            user_fea_dict = {}
            for node_ID in tqdm(one_DAG.nodes):
                att_list = one_DAG.nodes.get(node_ID)['att']
                user_fea_dict[node_ID] = []

                if float(att_list[2]) > 0 and float(att_list[3]) > 0:
                    user_fea_dict[node_ID].append(float(att_list[3]) / float(att_list[2]))
                else:
                    user_fea_dict[node_ID].append(0)

                if att_list[4] == "True":
                    user_fea_dict[node_ID].append(1)
                else:
                    user_fea_dict[node_ID].append(0)

                if att_list[1] == '-1.111111111' or att_list[1] == '-999' or (att_list[1] is None):
                    user_fea_dict[node_ID].append(2006)
                else:
                    f = convert_date_string(att_list[1])
                    if f is None:
                        user_fea_dict[node_ID].append(2006)
                    else:
                        user_fea_dict[node_ID].append(f)

                if (att_list[3] is None) or (float(att_list[3]) < 0):
                    user_fea_dict[node_ID].append(0)
                else:
                    user_fea_dict[node_ID].append(float(att_list[3]))

                if (att_list[2] is None) or (float(att_list[2]) < 0):
                    user_fea_dict[node_ID].append(0)
                else:
                    user_fea_dict[node_ID].append(float(att_list[2]))

                if (att_list[0] is None) or (float(att_list[0]) < 0):
                    user_fea_dict[node_ID].append(0)
                else:
                    user_fea_dict[node_ID].append(float(att_list[0]))

                one_DAG.nodes.get(node_ID)['profile_list'] = user_fea_dict[node_ID]

            
            
            feature_values = [[] for _ in range(6)]

            
            for _, data in tqdm(one_DAG.nodes(data=True)):
                profile_list = data['profile_list']
                for i, feature in enumerate(profile_list):
                    feature_values[i].append(feature)

            
            min_values = [min(values) for values in feature_values]
            max_values = [max(values) for values in feature_values]

            
            for _, data in tqdm(one_DAG.nodes(data=True)):
                profile_list = data['profile_list']
                normalized_profile = [
                    (feature - min_values[i]) / (max_values[i] - min_values[i]) if max_values[i] != min_values[
                        i] else 0.0 for i, feature in enumerate(profile_list)]
                data['norm_profile_list'] = normalized_profile

                data['inf'] = 0.0001 + norm_info_ordering[0] * normalized_profile[0] + \
                                                 norm_info_ordering[1] * normalized_profile[1] + \
                                                 norm_info_ordering[2] * normalized_profile[2] + \
                                                 norm_info_ordering[3] * normalized_profile[3] + \
                                                 norm_info_ordering[4] * normalized_profile[4] + \
                                                 norm_info_ordering[5] * normalized_profile[5]
            
            return one_DAG

        union_graph = twitter16.data['largest_component']
        union_graph_inf = importance_comp(union_graph)


        rumor_cascades = []
        for i, one_DAG in enumerate(tqdm(twitter16.data['propagation_DAG'])):
            y = twitter16.data['Fake_or_True'][i]
            if y == 'true':
                rumor_cascades.append(one_DAG)

    return rumor_cascades, union_graph_inf
