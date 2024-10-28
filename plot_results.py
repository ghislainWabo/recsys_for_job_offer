# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:06:25 2018

@author: NZEKON
"""

import os
import sys
import ast
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import six

from operator import itemgetter

file_dir = os.path.dirname(os.path.abspath(__file__))
xgboost = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(xgboost, "out")


def best_performance_allfile(in_dirname, bdname, metric_list):
    in_rep = os.path.join(out, in_dirname)
    in_filename = os.path.join(in_rep, bdname+".txt")
    infile = open(in_filename, 'r')
    first_line = infile.readline()
    file_results_dict = ast.literal_eval(first_line.strip())

    best_perform = {}
    for metric in metric_list:
        best_perform[metric] = {}
        for recsys_name in file_results_dict.keys():
            best_perform[metric][recsys_name] = file_results_dict[recsys_name][metric]
    #print(best_perform)
    return best_perform

def bd_best_performances_and_imp_for_all_recsys_type(in_dirname, bdname, metric_list, recsys_dict):
    all_best_perform = best_performance_allfile(in_dirname, bdname, metric_list)
    #print(all_best_perform)
    recsys_list = []
    bloc_nb_rs = 0
    for bloc in recsys_dict.keys():
        recsys_list.extend(recsys_dict[bloc])
        bloc_nb_rs = len(recsys_dict[bloc])

    tab_perform = np.zeros(shape=(len(recsys_list), 2 * len(metric_list)))
    id_bloc = -1
    for bloc in recsys_dict.keys():
        id_bloc += 1
        for metric_id in range(len(metric_list)):
            metric = metric_list[metric_id]
            base_perform = round(100.0 * all_best_perform[metric][recsys_id_to_name[recsys_dict[bloc][0]]], 2)
            for id_rs_in_bloc in range(len(recsys_dict[bloc])):
                i = id_bloc * bloc_nb_rs + id_rs_in_bloc
                j = 2 * metric_id
                rs_perform = round(100.0 * all_best_perform[metric][recsys_id_to_name[recsys_dict[bloc][id_rs_in_bloc]]], 2)
                if rs_perform > 99.9 or rs_perform < -9.99:
                    rs_perform = round(1.0 * rs_perform, 0)
                #print(id_bloc, i, j, base_perform, rs_perform)
                imp = 0.0
                if base_perform > 0.0:
                    imp = round((100.0 * (rs_perform - base_perform))/(1.0 * base_perform), 1)
                    if imp > 99.99 or imp < -9.99:
                        imp = int(imp//1)
                elif id_rs_in_bloc != 0 and  rs_perform > 0.0:
                    imp = infini_val
                tab_perform[i][j] = rs_perform
                tab_perform[i][j + 1] = imp
    return tab_perform

def plot_bd_best_performances_and_imp_for_recsys_type_list(in_dirname, bdname, metric_dict, recsys_dict):
    metric_list, col_labels, col_widths = [], [], []
    for metric_type_label in metric_dict.keys():
        metric_list.extend(list(metric_dict[metric_type_label].keys()))
        for metric_label in metric_dict[metric_type_label].values():
            col_labels.extend([metric_label, 'imp.'])
            col_widths.extend([0.5, 0.3])
    tab_perform = bd_best_performances_and_imp_for_all_recsys_type(in_dirname, bdname, metric_list, recsys_dict)
    nb_rs = len(recsys_dict) * 4
    data = np.zeros(shape = (nb_rs, 2 * len(metric_list) + 1))
    data_best = np.zeros(shape = (len(recsys_dict), 2 * len(metric_list) + 1))
    data_best_row_labels = []
    row_iter, row_labels = 0, []
    #print(tab_perform)
    #print(tab_perform)
    recsys_list = []
    id_bloc = -1
    for bloc in recsys_dict.keys():
        id_bloc += 1
        recsys_list.extend(recsys_dict[bloc])
        data_best_row_labels.append(bloc)
    for ii in range(len(tab_perform)):
        for jj in range(len(tab_perform[0])):
             data[ii][jj+1] = tab_perform[ii][jj]
        name_rs = recsys_id_to_name[recsys_list[ii]]
        row_labels.append(name_rs)
    id_bloc = -1
    for bloc in recsys_dict.keys():
        id_bloc += 1
        bloc_nb_rs = len(recsys_dict[bloc])
        for id_metric in range(len(metric_list)):
            bloc_vals = []
            for kk in range(bloc_nb_rs):
                bloc_vals.append(data[bloc_nb_rs*id_bloc+kk][1 + 2*id_metric])
            bloc_max = max(bloc_vals)
            nb_val_max = 0
            for kk in range(len(bloc_vals)):
                if bloc_vals[kk] == bloc_max:
                    nb_val_max += 1
            if bloc_vals[0] == bloc_max and nb_val_max == 1:
                data_best[id_bloc][1 + 2*id_metric] = 0.0
                data_best[id_bloc][2 + 2 * id_metric] = data[bloc_nb_rs*id_bloc+0][2 + 2*id_metric]
            elif bloc_vals[0] == bloc_max and nb_val_max > 1:
                data_best[id_bloc][1 + 2*id_metric] = -1.0
                data_best[id_bloc][2 + 2 * id_metric] = data[bloc_nb_rs*id_bloc+0][2 + 2*id_metric]
            elif bloc_vals[1] == bloc_max:
                data_best[id_bloc][1 + 2*id_metric] = 1.0
                data_best[id_bloc][2 + 2 * id_metric] = data[bloc_nb_rs*id_bloc+1][2 + 2*id_metric]
            elif bloc_vals[2] == bloc_max:
                data_best[id_bloc][1 + 2*id_metric] = 2.0
                data_best[id_bloc][2 + 2 * id_metric] = data[bloc_nb_rs*id_bloc+2][2 + 2*id_metric]
            elif bloc_vals[3] == bloc_max:
                data_best[id_bloc][1 + 2*id_metric] = 3.0
                data_best[id_bloc][2 + 2 * id_metric] = data[bloc_nb_rs*id_bloc+3][2 + 2*id_metric]

    fig_w, fig_h = 6, (1 + 1 * len(recsys_dict))/2.0 #(1 + 1.5 * len(recsys_dict))/2.0
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=data, colLabels=[bdname]+col_labels, loc='center', bbox=[-0.1, -0.05, 1.2, 1])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6.3)#(6.3)
    cellDict = the_table.get_celld()

    # fusioner les entêtes des colonnes pour que la métrique soit au dessus de la valeur et de imp (amélioration)
    # le positionnement des éléments pour avoir l'image souhaitée repose beaucoup sur le paramètre bbox
    header_metric_label = plt.table(cellText=[list(metric_dict.keys())], cellLoc='center', bbox=[0.067, 0.95, 1.033, 0.1])
    header_metric_label.auto_set_font_size(False)
    header_metric_label.set_fontsize(7)

    height_c = 0.05
    nb_row, nb_col = len(data)+1, len(data[0])
    width_c, width_c_imp = 0.062, 0.063
    row_labels = [bdname] + row_labels
    #print('row_labels',len(row_labels))
    for i in range(nb_row):
        #print(i)
        cellDict[(i, 0)]._text.set_text(row_labels[i])
        cellDict[(i, 0)]._loc = 'left'
        cellDict[(i, 0)].set_width(0.185)
        cellDict[(i, 0)].set_height(height_c)
        cellDict[(i, 0)].set_linewidth(0)
        cellDict[(i, 0)].set_fontsize(5.5)
        for j in range(nb_col-1):
            cellDict[(i, j+1)]._loc = 'center'
            cellDict[(i, j+1)].set_width(width_c)
            cellDict[(i, j+1)].set_height(height_c)
            cellDict[(i, j+1)].set_linewidth(0)
            if j%2 == 1:
                cellDict[(i, j+1)].set_width(width_c_imp)
            #if i%4 != 0:
            #    cellDict[(i, j+1)].visible_edges = 'vertical'#'horizontal' #set_linewidth(0)

    for i in range(len(data)):
        for j in range(len(data[0])-1):
            if j%2 == 1:
                if data[i, j+1] == 0.0:
                    cellDict[(i + 1, j+1)]._text.set_text("-")
                if data[i, j+1] < -9.99 or data[i, j+1] > 99.99:
                    cellDict[(i + 1, j + 1)]._text.set_text(str(int(data[i, j+1]//1)))
                if data[i, j+1] < 0.0:
                    cellDict[(i + 1, j + 1)]._text.set_color('red')
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_rouge_leger)
                    cellDict[(i + 1, j)].set_facecolor(coul_rouge_leger)
                if data[i, j+1] != infini_val and data[i, j+1] > 0.0:
                    cellDict[(i + 1, j + 1)]._text.set_color('blue')
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_bleu_leger)
                    cellDict[(i + 1, j)].set_facecolor(coul_bleu_leger)
                if data[i, j+1] == infini_val:
                    cellDict[(i + 1, j + 1)]._text.set_text(infini_txt)
                    cellDict[(i + 1, j + 1)]._text.set_color('blue')
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_bleu_leger)
                    cellDict[(i + 1, j)].set_facecolor(coul_bleu_leger)
                id_bloc = int(i//4)
                vals_bloc = [data[id_bloc*4, j], data[id_bloc*4+1, j], data[id_bloc*4+2, j], data[id_bloc*4+3, j]]
                #print(id_bloc, i, j, data[i, j], vals_bloc)
                max_bloc = max(vals_bloc)
                nb_egal_max = 0
                for val_bloc in vals_bloc:
                    if val_bloc == max_bloc:
                        nb_egal_max += 1
                if nb_egal_max == 1 and data[i, j] == max_bloc:
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_vert_leger)
                    cellDict[(i + 1, j)].set_facecolor(coul_vert_leger)

    # pour la zone du nom du jeu de données
    header_bdname_label = plt.table(cellText=[[bdname_dict[bdname]]], cellLoc='center', bbox=[-0.103, 0.895, 0.171, 0.155])
    header_bdname_label.auto_set_font_size(False)
    header_bdname_label.set_fontsize(5.5)

    # pour les traits séparateurs horizontaux des blocs
    width_hline = 1.2
    height_hline = 0.0001
    # pour les traits séparateurs horizontaux des blocs
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.1, -0.05, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.1, 0.19, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.1, 0.425, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.1, 0.66, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.1, 0.895, width_hline, height_hline])

    # premiere vague des traits verticaux
    add_h_list = {-1: -0.055, 0: 0, 1: -0.003, 2: -0.004, 4: -0.009, 5: -0.011, 7: -0.012, 8: -0.014, 9: -0.012}
    for kkk in [-1, 0, 1, 2, 4, 5, 7, 8, 9]:
        add_h = add_h_list[kkk]
        plt.table(cellText=[['']], cellLoc='center', bbox=[0.068 + kkk * 0.116 + add_h, -0.05, 0.0001, 1])

    # separateur verticaux
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[0.409, -0.05, 0.0045, 1])
    separateur_horizontal_celld = separateur_horizontal.get_celld()
    separateur_horizontal_celld[(0, 0)].set_facecolor('white')
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[0.753, -0.05, 0.0045, 1])
    separateur_horizontal_celld = separateur_horizontal.get_celld()
    separateur_horizontal_celld[(0, 0)].set_facecolor('white')

    timestamp = int((time.time()) // 1)
    in_rep = os.path.join(out, in_dirname)
    img_name = os.path.join(in_rep, in_dirname + "-plot-" + str(bdname) + "-" + str(timestamp) + ".pdf")

    plt.savefig(img_name, format='pdf', pad_inches=0)
    plt.close()
    plt.ioff()

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    #
    # Les meilleurs uniquement
    #
    ####################################################################################################################
    ####################################################################################################################
    #
    fig_w, fig_h = 6, (1 + 0.85 * len(recsys_dict)) / 5.25  # (1 + 1.5 * len(recsys_dict))/2.0
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=data_best, colLabels=[''] + col_labels, loc='center', bbox=[-0.1, -0.05, 1.2, 1])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6.3)
    cellDict = the_table.get_celld()

    # fusioner les entêtes des colonnes pour que la métrique soit au dessus de la valeur et de imp (amélioration)
    # le positionnement des éléments pour avoir l'image souhaitée repose beaucoup sur le paramètre bbox
    # bbox=[x_abcisse, y_ordonnée, largeur, hauteur]
    header_metric_label = plt.table(cellText=[list(metric_dict.keys())], cellLoc='center',  bbox=[0.039, 0.94, 1.061, 0.2])
    header_metric_label.auto_set_font_size(False)
    header_metric_label.set_fontsize(6)

    height_c = 0.05
    nb_row, nb_col = len(data_best) + 1, len(data_best[0])
    width_c, width_c_imp = 0.062, 0.063
    row_labels = [''] + data_best_row_labels
    for i in range(nb_row):
        cellDict[(i, 0)]._text.set_text(row_labels[i])
        cellDict[(i, 0)]._loc = 'right'
        cellDict[(i, 0)].set_width(0.15)
        cellDict[(i, 0)].set_height(height_c)
        cellDict[(i, 0)].set_linewidth(0)
        for j in range(nb_col - 1):
            cellDict[(i, j + 1)]._loc = 'center'
            cellDict[(i, j + 1)].set_width(width_c)
            cellDict[(i, j + 1)].set_height(height_c)
            cellDict[(i, j + 1)].set_linewidth(0)
            if j % 2 == 1:
                cellDict[(i, j + 1)].set_width(width_c_imp)

    for i in range(len(data_best)):
        for j in range(len(data_best[0]) - 1):
            if j % 2 == 0:
                if data_best[i, j + 2] != infini_val and (data_best[i, j + 2] < -9.99 or data_best[i, j + 2] > 99.99):
                    cellDict[(i + 1, j + 2)]._text.set_text(str(int(data_best[i, j+2]//1)))
                elif data_best[i, j + 2] == infini_val:
                    cellDict[(i + 1, j + 2)]._text.set_text(infini_txt)
                if data_best[i, j + 1] == 0.0:
                    cellDict[(i + 1, j + 1)]._text.set_text("-")
                    cellDict[(i + 1, j + 2)]._text.set_text(">")
                    cellDict[(i + 1, j + 2)]._text.set_color('red')
                if data_best[i, j + 1] == -1.0:
                    cellDict[(i + 1, j + 1)]._text.set_text("-")
                    cellDict[(i + 1, j + 2)]._text.set_text("==")
                elif data_best[i, j + 1] == 1.0:
                    cellDict[(i + 1, j + 1)]._text.set_text("OC")
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_bleu_leger)
                    cellDict[(i + 1, j + 2)].set_facecolor(coul_bleu_leger)
                    cellDict[(i + 1, j + 2)]._text.set_color('blue')
                    cellDict[(i + 1, j + 1)].set_fontsize(4)
                elif data_best[i, j + 1] == 2.0:
                    cellDict[(i + 1, j + 1)]._text.set_text("ZIPF")
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_vert_leger)
                    cellDict[(i + 1, j + 2)].set_facecolor(coul_vert_leger)
                    cellDict[(i + 1, j + 2)]._text.set_color('blue')
                    cellDict[(i + 1, j + 1)].set_fontsize(4)
                elif data_best[i, j + 1] == 3.0:
                    cellDict[(i + 1, j + 1)]._text.set_text("OWDC")
                    cellDict[(i + 1, j + 1)].set_facecolor(coul_jaune_leger)
                    cellDict[(i + 1, j + 2)].set_facecolor(coul_jaune_leger)
                    cellDict[(i + 1, j + 2)]._text.set_color('blue')
                    cellDict[(i + 1, j + 1)].set_fontsize(4)

    header_bdname_label = plt.table(cellText=[[bdname_dict[bdname]]], cellLoc='center', bbox=[-0.159, -0.05, 0.095, 0.805])#bbox=[-0.1, 0.655, 0.141, 0.49])
    header_bdname_label.auto_set_font_size(False)
    header_bdname_label.set_fontsize(5.5)

    # pour les traits séparateurs horizontaux des blocs
    width_hline, height_hline = 1.16, 0.0001
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.062, -0.05, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.062, 0.155, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.062, 0.355, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.062, 0.555, width_hline, height_hline])
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[-0.062, 0.755, width_hline, height_hline])


    # premiere vague des traits verticaux
    height_sep = 0.99
    add_h_list = {0:-0.002, 1:0, 2:0.002, 4:0.004, 5:0.006, 7:0.009, 8:0.013, 9:0.0149}
    for kkk in [0, 1, 2, 4, 5, 7, 8, 9]:
        add_h = add_h_list[kkk]
        plt.table(cellText=[['']], cellLoc='center', bbox=[0.041 + kkk * 0.116 + add_h, -0.05, 0.0001, height_sep])

    # separateurs verticaux
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[0.390, -0.05, 0.005, height_sep])
    separateur_horizontal_celld = separateur_horizontal.get_celld()
    separateur_horizontal_celld[(0, 0)].set_facecolor('white')
    separateur_horizontal = plt.table(cellText=[['']], cellLoc='center', bbox=[0.743, -0.05, 0.005, height_sep])
    separateur_horizontal_celld = separateur_horizontal.get_celld()
    separateur_horizontal_celld[(0, 0)].set_facecolor('white')

    timestamp = int((time.time()) // 1)
    in_rep = os.path.join(out, in_dirname)
    img_name = os.path.join(in_rep, in_dirname + "-plot-best-" + str(bdname) + "-" + str(timestamp) + ".pdf")

    plt.savefig(img_name, format='pdf', pad_inches=0)
    plt.close()
    plt.ioff()

########################################################################################################################
########################################################################################################################

coul_vert_leger = '#e4ffd4'
coul_bleu_leger = '#d4e4ff'
coul_rouge_leger = '#ffe4d4'
coul_jaune_leger = '#ffffd4'

bdname_dict = {'minajobs': 'MINAJOBS', 'monster':'MONSTER', 'nigham':'NIGHAM'}

algos = ['DT', 'NB']
vectorization = ['TFIDF', 'D2V']
transform = ['BASIC', 'OC', 'ZIPF', 'OWDC']
recsys_name_to_id = {}
recsys_id_to_name = {}
recsys_dict = {}
id_recsys_ = 0
for algo in algos:
    for vecto in vectorization:
        recsys_dict[algo+'-'+vecto] = []
        for trans in transform:
            recsys_name_to_id[algo+'-'+vecto+'-'+trans] = id_recsys_
            recsys_id_to_name[id_recsys_] = algo + '-' + vecto + '-' + trans
            recsys_dict[algo+'-'+vecto].append(id_recsys_)
            id_recsys_ += 1

infini_val = 999999
infini_txt = r'$+\infty$'

#top_n_list = [3, 5, 10]
top_n_list = [1, 2, 3]
metric_list = ['Precision', 'MRR', 'Recall']
metric_dict_name = {'MAP':['map','M'], 'MRR':['mrr','R'], 'F1-score':['f1','F'], 'Precision':['prec','P'], 'Recall':['recall', 'C'], 'Accuracy':['acc', 'A']}
metric_dict = {}
for metric in metric_list:
    metric_dict[metric] = {}
    for top in top_n_list:
        metric_dict[metric][metric_dict_name[metric][0]+'@'+str(top)] = metric_dict_name[metric][1]+'@'+str(top)


def main(in_dirname, bdname):
    print(metric_dict)
    print(recsys_dict)
    print(recsys_name_to_id)
    print(recsys_id_to_name, '\n')
    plot_bd_best_performances_and_imp_for_recsys_type_list(in_dirname, bdname, metric_dict, recsys_dict)

if __name__ == "__main__":
    main('cari-results', 'minajobs')
    main('cari-results', 'monster')
    main('cari-results', 'nigham')

    print('END !!!')

