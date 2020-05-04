import matplotlib.pyplot as plt
import numpy as np

feat_x2_0_inter = np.array([0.34256017,	0.31460512,	0.27144957,	0.24856855,	0.22719437])
feat_x2_0_inter_ste = np.array([0.004312400818,	0.003773893416,	0.002982698083,	0.00233306855,	0.002138502002])
feat_x2_1_inter = np.array([0.35977444,	0.3302486,	0.2858478,	0.26235148,	0.23809971])
feat_x2_1_inter_ste = np.array([0.004517972469,	0.004006842971,	0.003241877258,	0.002627826631,	0.002335606366])
feat_x2_2_inter = np.array([0.387529,	0.35519564,	0.30843478,	0.28350833,	0.2540046])
feat_x2_2_inter_ste = np.array([0.004769945741,	0.00426604718,	0.00358707428,	0.003047236502,	0.002536125779])
feat_x2_3_inter = np.array([0.4165158,	0.38317177,	0.33215567,	0.3049404,	0.27241272])
feat_x2_3_inter_ste = np.array([0.005206080675,	0.004714588821,	0.003986877203,	0.003410176337,	0.002820271552])
feat_x2_4_inter = np.array([0.44910964,	0.41258878,	0.35587674,	0.3253371,	0.29092044])
feat_x2_4_inter_ste = np.array([0.005780712962,	0.005148547292,	0.004204738736,	0.003574062288,	0.003055044413])
ERM = np.array([0.37845063, 0.37845063, 0.37845063, 0.37845063, 0.37845063])
ERM_ste = np.array([0.004980756044, 0.004980756044, 0.004980756044, 0.004980756044, 0.004980756044])

ERM_x2_only = np.array([0.22802237, 0.22802237, 0.22802237, 0.22802237, 0.22802237])
ERM_ste_x2_only = np.array([0.0021790754795074463, 0.0021790754795074463, 0.0021790754795074463, 0.0021790754795074463, 0.0021790754795074463])

x = [1, 2, 3, 4, 5]

fig, ax = plt.subplots()
plt.plot(x, ERM, label='ERM')
# plt.fill_between(x, ERM - ERM_ste, ERM + ERM_ste, alpha=0.1)
markers, caps, bars = ax.errorbar(x, feat_x2_0_inter, yerr=feat_x2_0_inter_ste, label='augmentation on 0 dims of $h_y$')
# plt.fill_between(x, feat_x2_0_inter - feat_x2_0_inter_ste, feat_x2_0_inter + feat_x2_0_inter_ste, alpha=0.1)
markers, caps, bars = ax.errorbar(x, feat_x2_1_inter, yerr=feat_x2_1_inter_ste, label='augmentation on 1 dim of $h_y$')
# plt.fill_between(x, feat_x2_1_inter - feat_x2_1_inter_ste, feat_x2_1_inter + feat_x2_1_inter_ste, alpha=0.1)
markers, caps, bars = ax.errorbar(x, feat_x2_2_inter, yerr=feat_x2_2_inter_ste, label='augmentation on 2 dims of $h_y$')
# plt.fill_between(x, feat_x2_2_inter - feat_x2_2_inter_ste, feat_x2_2_inter + feat_x2_2_inter_ste, alpha=0.1)
markers, caps, bars = ax.errorbar(x, feat_x2_3_inter, yerr=feat_x2_3_inter_ste, label='augmentation on 3 dims of $h_y$')
# plt.fill_between(x, feat_x2_3_inter - feat_x2_3_inter_ste, feat_x2_3_inter + feat_x2_3_inter_ste, alpha=0.1)
markers, caps, bars = ax.errorbar(x, feat_x2_4_inter, yerr=feat_x2_4_inter_ste, label='augmentation on 4 dims of $h_y$')
plt.plot(x, ERM_x2_only, label='ERM using only $h_y$')
# plt.fill_between(x, feat_x2_4_inter - feat_x2_4_inter_ste, feat_x2_4_inter + feat_x2_4_inter_ste, alpha=0.1)
plt.xticks(x) # Set locations and labels
plt.legend()
plt.ylabel('$MSE$')
plt.xlabel('num of dims of $h_d$ w/ augmentation')
plt.savefig('toy_data_comparison.png', bbox_inches='tight', dpi=300)