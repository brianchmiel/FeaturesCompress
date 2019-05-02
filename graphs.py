import matplotlib.pyplot as plt
import torch
import numpy as np
from entropy import shannon_entropy
import random

if __name__ == '__main__':

########################
    #EigenValues test

    # ratio =  np.linspace(0.40, 0.95, 12)
    # resnet18 = [0.05572150735294118,	0.06767003676470588,	0.0818014705882353,	0.09823069852941177,	0.11592371323529412,	0.14039522058823528,	0.16716452205882354,	0.19852941176470587,	0.24103860294117646,	0.29733455882352944,	0.37706801470588236,	0.5098805147]
    # resnet101 = [0.03952088647959184,	0.047951211734693876,	0.05655094068877551,	0.06819993622448979,	0.0812240911989796,	0.09598214285714286,	0.11446707589285714,	0.1365593112244898,	0.16652383609693877,	0.20715082908163265,	0.2683055644132653,	0.3822245695]
    # resnet50 = [0.0437548828125,	0.054443359375,	0.067783203125,	0.0822265625,	0.1008447265625,	0.1219873046875,	0.1470361328125,	0.177548828125,	0.2147705078125,	0.2629638671875,	0.33115234375,	0.4452099609]
    # mobilenet = [0.02524801637046039,	0.03125248114644949,	0.03709077464217054,	0.04543154849338212,	0.05509672763956977,	0.06626736216047513,	0.07873760065662541,	0.09629960520791687,	0.11984375186064945,	0.15132688872316585,	0.1993303620988237,	0.2852653853]
    # inception = [0.056502502503250035,	0.06948692473196523,	0.08258691297576545,	0.09963826790075511,	0.11990406765463821,	0.1433858951117764,	0.1723499270750487,	0.2082153984365311,	0.2527957241585914,	0.31153432739541886,	0.3950964164860705,	0.5326771562]
    #
    # plt.plot(ratio, resnet18)
    # plt.plot(ratio, resnet50)
    # plt.plot(ratio, resnet101)
    # plt.plot(ratio, mobilenet)
    # plt.plot(ratio, inception)
    # plt.gca().legend(('ResNet18','ResNet50','ResNet101','MobileNetV2','InceptionV3'))
    # plt.xlabel('EigenValues sum ratio')
    # plt.ylabel('EigenValues index ratio')
    # plt.show()
########################

# #Windows shape

    # W11C_Acc = [62.5,65.65,67.7,68.79,69.43,69.73]
    # W11C_Entropy = [2.2,2.38,2.75,3.18,3.63,4.1]
    # W11C_MSE = [0.004,0.0029,0.0015,0.0008,0.0004,0.0002]
    #
    # W22C_4_Acc = [55.72,62.24,66.2,67.9,68.772,69.214,69.4]
    # W22C_4_Entropy = [2.2,2.61,3,3.48,3.94,4.41,4.91]
    # W22C_4_MSE = [0.0062,0.0032,0.0016,0.0008,0.0004,0.0002,0.0001]
    #
    # W44C_16_Acc = [50.616,60.59,64.168,66.86,68.3,69.2,69.4]
    # W44C_16_Entropy = [2,2.36,2.76,3.19,3.6,4,4.2]
    # W44C_16_MSE = [0.0135,0.0071,0.0037,0.002,0.001,0.0008,0.0004]
    #
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    #
    # ax1.set_xlabel('Avg bits per value')
    # ax1.set_ylabel('Accurracy (%)')
    # ax1.set_ylim([60,70])
    # ax1.set_xlim([2, 4.5])
    # ax2.set_xlim([2, 4.5])
    # ax2.set_ylabel('MSE')
    #
    # ax1.plot(W11C_Entropy, W11C_Acc, color='blue')
    # ax2.plot(W11C_Entropy, W11C_MSE, color='b',linestyle='--')
    #
    # ax1.plot(W22C_4_Entropy, W22C_4_Acc, color='red')
    # ax2.plot(W22C_4_Entropy, W22C_4_MSE, color='red',linestyle='--')
    #
    # ax1.plot(W44C_16_Entropy, W44C_16_Acc, color='green')
    # ax2.plot(W44C_16_Entropy, W44C_16_MSE, color='green',linestyle='--')
    #
    # ax1.plot(torch.linspace(2,5,100).numpy(),torch.tensor(69.73).repeat(100).numpy(), color= 'black', linestyle = '-.')
    #
    # ax1.legend(('1*1*C - Accuracy', '2*2*(C/4) - Accuracy',
    #             '4*4*(C/16) - Accuracy', 'Baseline' ),loc='center right')
    #
    # ax2.legend(('1*1*C - MSE', '2*2*(C/4) - MSE'
    #                   , '4*4*(C/16) - MSE',),loc='center right')
    # plt.show()

########################


# #Windows size
    #
    # W11C_Acc = [62.5,65.65,67.7,68.79,69.43,69.73]
    # W11C_Entropy = [2.2,2.38,2.75,3.18,3.63,4.1]
    # W11C_MSE = [0.004,0.0029,0.0015,0.0008,0.0004,0.0002]
    #
    # W11C_2_Acc = [67.086,68.488,69.026,69.448,69.5,69.536,69.662]
    # W11C_2_Entropy = [3.25,3.73,4.21,4.72,5.24,5.71,6.2]
    # W11C_2_MSE = [0.0019,0.001,0.0005,0.0002,0.00015,0.00012,0.0001]
    #
    # W11C_4_Acc = [67.888,68.798,69.316,69.416,69.6,69.612,69.628]
    # W11C_4_Entropy = [3.76,4.26,4.79,5.2,5.79,6.3,7.3]
    # W11C_4_MSE = [0.0014,0.0007,0.0004,0.0002,0.00015,0.00012,0.0001]
    #
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    #
    # ax1.set_xlabel('Avg bits per value')
    # ax1.set_ylabel('Accurracy (%)')
    # ax1.set_ylim([62,70])
    # ax1.set_xlim([2, 6])
    # ax2.set_xlim([2, 6])
    # ax2.set_ylabel('MSE')
    #
    # ax1.plot(W11C_Entropy, W11C_Acc, color='blue')
    # ax2.plot(W11C_Entropy, W11C_MSE, color='b',linestyle='--')
    #
    # ax1.plot(W11C_2_Entropy, W11C_2_Acc, color='red')
    # ax2.plot(W11C_2_Entropy, W11C_2_MSE, color='red',linestyle='--')
    #
    # ax1.plot(W11C_4_Entropy, W11C_4_Acc, color='green')
    # ax2.plot(W11C_4_Entropy, W11C_4_MSE, color='green',linestyle='--')
    #
    # ax1.plot(torch.linspace(2,6,100).numpy(),torch.tensor(69.73).repeat(100).numpy(), color= 'black', linestyle = '-.')
    #
    # ax1.legend(('1*1*C - Accuracy', '1*1*(C/2) - Accuracy',
    #              '1*1*(C/4) - Accuracy', 'Baseline' ),loc='center right')
    #
    # ax2.legend(('1*1*C - MSE', '1*1*(C/2) - MSE'
    #                   , '1*1*(C/4) - MSE',),loc='center right')
    # plt.show()

########################


# #Per layer
#
#     entr = [3.28279519081, 4.06368255615, 5.34731769562, 5.44142913818, 5.79445648193,5.44872093201, 5.66347932816,5.96365451813,
#     5.52262210846,5.01675462723,5.66567325592,5.02026891708,5.14488554001,4.1341047287,4.84526491165,3.9473567009,4.24032068253]
#
#     layer = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#     plt.bar(layer,entr)
#     plt.xlim(0, 18)
#     plt.xlabel('Layer number')
#     plt.ylabel('avg number of bits per value')
#     plt.show()

########################

# #memory - computational tradeoff
#
#     entr = [4.15,4.87,5.5,6.1,6.6,8.2]
#     comp = [1,0.97,0.95,0.93,0.88,0.83]
#
#     plt.scatter(comp, entr)
#     plt.xlabel('Computational Complexity Ratio')
#     plt.ylabel('Memory Bandwidth Complexity')
#     plt.show()

########################

# Resnet 18

    w8_act = [67.7,68.7,69.25,69.5,69.73]
    w8_entr = [2.75,3.1,3.5,4,4.23]
    w8_huff = [2.83,3.18,3.53,4,4.24]

    w4_act = [67,68.7,69,69.3,69.7]
    w4_entr = [2.7,3.2,3.65,4,4.3]
    w4_huff = [2.83,3.28,3.70,4.04,4.32]

    baseline = 69.76

    plt.plot(w8_entr, w8_act, color = 'blue')
    plt.plot(w4_entr, w4_act, color = 'red')
    plt.plot(w8_huff, w8_act, color='blue',linestyle='--')
    plt.plot(w4_huff, w4_act, color='red',linestyle='--')
    plt.plot(torch.linspace(2,6,100).numpy(),torch.tensor(baseline).repeat(100).numpy(), color= 'black', linestyle = '-.')

    plt.gca().legend(('Theoretical Entropy - 8 bit Weights',
                      'Theoretical Entropy - 4 bit Weights','Huffman - 8 bit Weights',
                      'Huffman - 4 bit Weights', 'Baseline'))


    plt.xlabel('Avg bits per value')
    plt.ylabel('Accuracy (%)')
    plt.xlim([2.5,4.5])
    plt.ylim([66,70])
    plt.title('ResNet18')
    plt.show()


# Resnet 50

    w8_act = [74.4,75.3,75.9,76.12]
    w8_entr = [2.7,3.2,4,4.15]
    w8_huff = [2.83,3.28,4.06,4.17]

    w4_act = [74,75,75.7,76.08]
    w4_entr = [2.7,3.2,4,4.25]
    w4_huff = [2.83,3.28,4.06,4.27]

    baseline = 76.12

    plt.plot(w8_entr, w8_act, color = 'blue')
    plt.plot(w4_entr, w4_act, color = 'red')
    plt.plot(w8_huff, w8_act, color='blue',linestyle='--')
    plt.plot(w4_huff, w4_act, color='red',linestyle='--')
    plt.plot(torch.linspace(2,6,100).numpy(),torch.tensor(baseline).repeat(100).numpy(), color= 'black', linestyle = '-.')

    plt.gca().legend(('Theoretical Entropy - 8 bit Weights',
                      'Theoretical Entropy - 4 bit Weights','Huffman - 8 bit Weights',
                      'Huffman - 4 bit Weights', 'Baseline'))


    plt.xlabel('Avg bits per value')
    plt.ylabel('Accuracy (%)')
    plt.xlim([2.5,4.5])
    plt.ylim([74,76.2])
    plt.title('ResNet50')
    plt.show()

# Resnet 101

    w8_act = [75,76.5,77,77.34]
    w8_entr = [2.7,3.2,3.3,3.9]
    w8_huff = [2.95,3.4,3.5,4]

    w4_act = [75,76,77,77.31]
    w4_entr = [2.7,3.3,3.6,4.1]
    w4_huff = [3,3.6,3.8,4.25]

    baseline = 77.37

    plt.plot(w8_entr, w8_act, color = 'blue')
    plt.plot(w4_entr, w4_act, color = 'red')
    plt.plot(w8_huff, w8_act, color='blue',linestyle='--')
    plt.plot(w4_huff, w4_act, color='red',linestyle='--')
    plt.plot(torch.linspace(2,6,100).numpy(),torch.tensor(baseline).repeat(100).numpy(), color= 'black', linestyle = '-.')

    plt.gca().legend(('Theoretical Entropy - 8 bit Weights',
                      'Theoretical Entropy - 4 bit Weights','Huffman - 8 bit Weights',
                      'Huffman - 4 bit Weights', 'Baseline'))


    plt.xlabel('Avg bits per value')
    plt.ylabel('Accuracy (%)')
    plt.xlim([2.6,4.2])
    plt.ylim([74,77.5])
    plt.title('ResNet101')
    plt.show()

# inception

    w8_act = [76.13,77,77.2,77.43]
    w8_entr =  [2.72,3.2,3.7,4.3]
    w8_huff = [3.2,3.43,3.9,4.34]

    w4_act = [76,76.9,77.1,77.4]
    w4_entr = [2.75,3.2,3.7,4.6]
    w4_huff = [3.2,3.43,4,4.66]

    baseline = 77.45

    plt.plot(w8_entr, w8_act, color = 'blue')
    plt.plot(w4_entr, w4_act, color = 'red')
    plt.plot(w8_huff, w8_act, color='blue',linestyle='--')
    plt.plot(w4_huff, w4_act, color='red',linestyle='--')
    plt.plot(torch.linspace(2,6,100).numpy(),torch.tensor(baseline).repeat(100).numpy(), color= 'black', linestyle = '-.')

    plt.gca().legend(('Theoretical Entropy - 8 bit Weights',
                      'Theoretical Entropy - 4 bit Weights','Huffman - 8 bit Weights',
                      'Huffman - 4 bit Weights', 'Baseline'))


    plt.xlabel('Avg bits per value')
    plt.ylabel('Accuracy (%)')
    plt.xlim([2.7,4.7])
    plt.ylim([74,77.5])
    plt.title('InceptionV3')
    plt.show()

# mobilenet

    w8_act = [63,70.7,71.4,71.77]
    w8_entr =  [2.33,3.1,3.25,3.8]
    w8_huff = [2.6,3.35,3.4,3.85]

    w4_act = [61,70.4,71.08,71.71]
    w4_entr = [2.33,3.1,3.4,4]
    w4_huff = [2.6,3.35,3.6,4.03]

    baseline = 71.8

    plt.plot(w8_entr, w8_act, color = 'blue')
    plt.plot(w4_entr, w4_act, color = 'red')
    plt.plot(w8_huff, w8_act, color='blue',linestyle='--')
    plt.plot(w4_huff, w4_act, color='red',linestyle='--')
    plt.plot(torch.linspace(2,6,100).numpy(),torch.tensor(baseline).repeat(100).numpy(), color= 'black', linestyle = '-.')

    plt.gca().legend(('Theoretical Entropy - 8 bit Weights',
                      'Theoretical Entropy - 4 bit Weights','Huffman - 8 bit Weights',
                      'Huffman - 4 bit Weights', 'Baseline'))


    plt.xlabel('Avg bits per value')
    plt.ylabel('Accuracy (%)')
    plt.xlim([2.6,4.1])
    plt.ylim([66,72])
    plt.title('MobileNetV2')
    plt.show()





#
#
# #Specific layer
#
#     entr = [7.33862400055,6.94230127335,6.80950307846,6.75260925293,6.20323133469,5.92172718048,5.62046194077,5.59032821655,5.4094247818,
#             5.23786592484,5.21044683456,5.02663326263,4.80640745163,4.66426897049,4.43449640274,4.36516714096,4.18917989731,4.01052713394,3.86077046394,
#             3.70227766037,3.57333827019,3.42303323746,3.31598758698,3.18312716484,3.07826852798,2.88044023514,2.76813364029,2.70250797272,2.60432267189,2.491492033,
#             2.39900994301,2.32703614235,2.24368429184,2.13318967819,2.01120448112,1.90284788609,1.77022087574,1.72623622417,1.63355207443,1.55765736103,1.46317970753,1.38078725338,
#             1.29158198833,1.19527924061,1.11342322826,1.01670300961,0.913911223412,0.835507035255,0.765761256218,0.685242831707,0.568300902843,0.545278072357,0.458970963955,0.393366247416,
#             0.341455698013,0.265832990408,0.199616760015,0.168380707502,0.111723564565,0.0962927937508,0.0638531446457,0.0269777886569,0.0195965357125,0.00488292565569]
#
#     channel = np.linspace(1,64,64)
#     plt.bar(channel,entr)
#     plt.xlim(0, 65)
#     plt.xlabel('channel number')
#     plt.ylabel('number of bits per value')
#     plt.show()
#
#



  #
  #
  #   def uniform_midtread_quantizer(x, max,min,bins):
  #       scale = bins / (max-min)
  #       xQ = np.round((x - min) * scale )
  #       return xQ, 1/scale
  #
  #
  #   def simulator(Val, NumOfbins):
  #
  #       Q = (max(Val) - min(Val)) / NumOfbins
  #       ValQ, scale = uniform_midtread_quantizer(Val, max(Val), min(Val), NumOfbins)
  #       entr_1D = shannon_entropy(torch.tensor(ValQ))
  #       ValQ = ValQ * scale + min(Val)
  #
  #       mse_1D = ((ValQ - Val) ** 2).mean()
  #       # simulations.append(mse)
  #       return [entr_1D, mse_1D, Q]
  #
  #
  #   Num_of_elements = 100000
  #   X = np.random.uniform(-0.5, 0.5, Num_of_elements)
  #  # X = np.random.normal(size=Num_of_elements, scale=1.0)
  #   Y = np.random.uniform(-4.0, 4.0, Num_of_elements)
  # #  Y = np.random.normal(size=Num_of_elements, scale=8.0)
  #   NumOfBins = 16
  #   Range = (0.0001, 10)
  #
  #
  #   simulations = []
  #   Ratio = []
  #   MSE = []
  #
  #   for i in range(Range[0], Range[1]):
  #       mse_X, Q_X = simulator(X, i)
  #       mse_Y, Q_Y = simulator(Y, NumOfBins - i)
  #       # simulations.append((i,NumOfBins-i,mse_X+mse_Y))
  #       simulations.append((Q_Y / Q_X, mse_X + mse_Y))
  #       Ratio.append(Q_Y / Q_X)
  #       MSE.append(mse_X + mse_Y)
  #   print(simulations)
  #   plt.plot(Ratio, MSE, 'b', linewidth=3, alpha=0.75)
  #
  #   # Y = np.random.uniform(-32.0, 32.0, Num_of_elements)
  #   # NumOfBins = 256
  #   # Range = 10
  #   #
  #   # for i in range(Range):
  #   #     mse_X, Q_X = simulator(X,i)
  #   #     mse_Y, Q_Y = simulator(Y,NumOfBins-i)
  #   #     # simulations.append((i,NumOfBins-i,mse_X+mse_Y))
  #   #     simulations.append((Q_Y/Q_X, mse_X + mse_Y))
  #   #     Ratio.append(Q_Y/Q_X)
  #   #     MSE.append(mse_X + mse_Y)
  #   # print(simulations)
  #   # plt.plot(Ratio, MSE, 'b', linewidth=3, alpha=0.75)
  #
  #
  #   plt.legend(('a/b=8', 'a/b=64'), loc=0)
  #   plt.ylabel('Mean Square Error', size=20)
  #   plt.xlabel('Delta_i/Delta_j', size=20)
  #
  #   plt.show()



    # import heapq
    # import os
    #
    #
    # class HeapNode:
    #     def __init__(self, char, freq):
    #         self.char = char
    #         self.freq = freq
    #         self.left = None
    #         self.right = None
    #
    #     def __cmp__(self, other):
    #         if (other == None):
    #             return -1
    #         if (not isinstance(other, HeapNode)):
    #             return -1
    #         return self.freq > other.freq
    #
    #
    # class HuffmanCoding:
    #     def __init__(self, path):
    #         self.path = path
    #         self.heap = []
    #         self.codes = {}
    #         self.reverse_mapping = {}
    #
    #     # functions for compression:
    #
    #     def make_frequency_dict(self, text):
    #         frequency = {}
    #         for character in text:
    #             if not character in frequency:
    #                 frequency[character] = 0
    #             frequency[character] += 1
    #         return frequency
    #
    #     def make_heap(self, frequency):
    #         for key in frequency:
    #             node = HeapNode(key, frequency[key])
    #             heapq.heappush(self.heap, node)
    #
    #     def merge_nodes(self):
    #         while (len(self.heap) > 1):
    #             node1 = heapq.heappop(self.heap)
    #             node2 = heapq.heappop(self.heap)
    #
    #             mergedName = '(' + node1.char + ',' + node2.char + ')'
    #             merged = HeapNode(mergedName, node1.freq + node2.freq)
    #             merged.left = node1
    #             merged.right = node2
    #
    #             heapq.heappush(self.heap, merged)
    #
    #     def make_codes_helper(self, root, current_code):
    #         if (root == None):
    #             return
    #
    #         if (root.char != None):
    #             self.codes[root.char] = current_code
    #             self.reverse_mapping[current_code] = root.char
    #             return
    #
    #         self.make_codes_helper(root.left, current_code + "0")
    #         self.make_codes_helper(root.right, current_code + "1")
    #
    #     def make_codes(self):
    #         root = heapq.heappop(self.heap)
    #         current_code = ""
    #         self.make_codes_helper(root, current_code)
    #
    #     def get_encoded_text(self, text):
    #         encoded_text = ""
    #         for character in text:
    #             encoded_text += self.codes[character]
    #         return encoded_text
    #
    #     def pad_encoded_text(self, encoded_text):
    #         extra_padding = 8 - len(encoded_text) % 8
    #         for i in range(extra_padding):
    #             encoded_text += "0"
    #
    #         padded_info = "{0:08b}".format(extra_padding)
    #         encoded_text = padded_info + encoded_text
    #         return encoded_text
    #
    #     def get_byte_array(self, padded_encoded_text):
    #         if (len(padded_encoded_text) % 8 != 0):
    #             print("Encoded text not padded properly")
    #             exit(0)
    #
    #         b = bytearray()
    #         for i in range(0, len(padded_encoded_text), 8):
    #             byte = padded_encoded_text[i:i + 8]
    #             b.append(int(byte, 2))
    #         return b
    #
    #     def compress(self):
    #         filename, file_extension = os.path.splitext(self.path)
    #         output_path = filename + ".bin"
    #
    #         with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
    #             text = file.read()
    #             text = text.rstrip()
    #
    #             frequency = self.make_frequency_dict(text)
    #             self.make_heap(frequency)
    #             self.merge_nodes()
    #             self.make_codes()
    #
    #             encoded_text = self.get_encoded_text(text)
    #             padded_encoded_text = self.pad_encoded_text(encoded_text)
    #
    #             b = self.get_byte_array(padded_encoded_text)
    #             output.write(bytes(b))
    #
    #
    #         print("Compressed")
    #         return output_path
    #
    #     def compressNew(self, frequency):
    #
    #             self.make_heap(frequency)
    #             self.merge_nodes()
    #
    #             self.make_codes()
    #
    #
    #
    #     """ functions for decompression: """
    #
    #
    #
    #     def remove_padding(self, padded_encoded_text):
    #         padded_info = padded_encoded_text[:8]
    #         extra_padding = int(padded_info, 2)
    #
    #         padded_encoded_text = padded_encoded_text[8:]
    #         encoded_text = padded_encoded_text[:-1 * extra_padding]
    #
    #         return encoded_text
    #
    #     def decode_text(self, encoded_text):
    #         current_code = ""
    #         decoded_text = ""
    #
    #         for bit in encoded_text:
    #             current_code += bit
    #             if (current_code in self.reverse_mapping):
    #                 character = self.reverse_mapping[current_code]
    #                 decoded_text += character
    #                 current_code = ""
    #
    #         return decoded_text
    #
    #     def decompress(self, input_path):
    #         filename, file_extension = os.path.splitext(self.path)
    #         output_path = filename + "_decompressed" + ".txt"
    #
    #         with open(input_path, 'rb') as file, open(output_path, 'w') as output:
    #             bit_string = ""
    #
    #             byte = file.read(1)
    #             while (byte != ""):
    #                 byte = ord(byte)
    #                 bits = bin(byte)[2:].rjust(8, '0')
    #                 bit_string += bits
    #                 byte = file.read(1)
    #
    #             encoded_text = self.remove_padding(bit_string)
    #
    #             decompressed_text = self.decode_text(encoded_text)
    #
    #             output.write(decompressed_text)
    #
    #         print("Decompressed")
    #         return output_path
    #
    #
    #
    # h = HuffmanCoding('home')
    # freqProj = {'0': 4.982461554448037e-09, '1': 0.0, '10': 3.4877231769314676e-08, '100': 0.0004950125585310161, '101': 0.0005623704637400806, '102': 0.0005977359833195806, '103': 0.000668412190862, '104': 0.0007490832358598709, '105': 0.0009091547690331936, '106': 0.0009904884500429034, '107': 0.0011191555531695485, '108': 0.0012847228208556771, '109': 0.0014705038629472256, '11': 2.989477110304506e-08, '110': 0.0017798847984522581, '111': 0.0021665687672793865, '112': 0.002746128709986806, '113': 0.0030957378912717104, '114': 0.003810168243944645, '115': 0.0047581614926457405, '116': 0.006316336337476969, '117': 0.00871968176215887, '118': 0.012302171438932419, '119': 0.018331527709960938, '12': 4.484215665456759e-08, '120': 0.03499671816825867, '121': 0.1418391913175583, '122': 0.6155323386192322, '123': 0.04488591104745865, '124': 0.020758744329214096, '125': 0.012524508871138096, '126': 0.008611741475760937, '127': 0.0063993241637945175, '128': 0.004657311365008354, '129': 0.003656170330941677, '13': 4.9824617320837206e-08, '130': 0.0029429856222122908, '131': 0.0024599211756139994, '132': 0.0020462123211473227, '133': 0.0017304538050666451, '134': 0.0015125159407034516, '135': 0.0012990572722628713, '136': 0.0011477499501779675, '137': 0.001017044996842742, '138': 0.0009077646536752582, '139': 0.0008033073390834033, '14': 3.4877231769314676e-08, '140': 0.0007164929411374032, '141': 0.000644516316242516, '142': 0.0006042978493496776, '143': 0.0005413544131442904, '144': 0.0005207619396969676, '145': 0.0004617944941855967, '146': 0.0004232053179293871, '147': 0.000400400604121387, '148': 0.00036859753890894353, '149': 0.00033913622610270977, '15': 5.480707798710682e-08, '150': 0.0003170489799231291, '151': 0.0002999242569785565, '152': 0.00028183794347569346, '153': 0.0002738061884883791, '154': 0.0002594716497696936, '155': 0.00024760840460658073, '156': 0.00023739934840705246, '157': 0.0002266073424834758, '158': 0.00022405631898436695, '159': 0.000219128662138246, '16': 5.978954220609012e-08, '160': 0.00020112204947508872, '161': 0.00019134646572638303, '162': 0.00017746532103046775, '163': 0.0001697674160823226, '164': 0.00016815309936646372, '165': 0.00017357899923808873, '166': 0.00017172552179545164, '167': 0.00016708187467884272, '168': 0.00016139190120156854, '169': 0.00015654895105399191, '17': 6.975446353862935e-08, '170': 0.00014937420201022178, '171': 0.0001432457793271169, '172': 0.00013433214917313308, '173': 0.00011877690849360079, '174': 0.000185477125342004, '175': 4.074159005540423e-05, '176': 3.2475683838129044e-05, '177': 3.066705176024698e-05, '178': 2.4827606466715224e-05, '179': 1.9800303562078625e-05, '18': 4.484215665456759e-08, '180': 1.6108298950712197e-05, '181': 1.4548788385582156e-05, '182': 1.5086894563864917e-05, '183': 1.4633490536652971e-05, '184': 1.3945910723123234e-05, '185': 1.1424785043345764e-05, '186': 1.1823382010334171e-05, '187': 1.218710167449899e-05, '188': 1.1992785402981099e-05, '189': 1.080197671399219e-05, '19': 7.971938487116859e-08, '190': 9.631098691897932e-06, '191': 8.52997436595615e-06, '192': 9.182676876662299e-06, '193': 8.111447641567793e-06, '194': 7.712850674579386e-06, '195': 7.359095889114542e-06, '196': 5.96898917137878e-06, '197': 5.540497568290448e-06, '198': 5.077128662378527e-06, '199': 4.653619271266507e-06, '2': 0.0, '20': 9.964923464167441e-08, '200': 4.2002152440545615e-06, '201': 4.394531060825102e-06, '202': 4.419443484948715e-06, '203': 4.489198090595892e-06, '204': 4.095583335583797e-06, '205': 3.68702171726909e-06, '206': 4.439373242348665e-06, '207': 6.736288469255669e-06, '208': 2.4922273951233365e-05, '209': 3.836495579889743e-06, '21': 1.4947384840979794e-07, '210': 2.610810042824596e-06, '211': 1.1559311587916454e-06, '212': 7.174745064730814e-07, '213': 5.381058940656658e-07, '214': 6.377550789693487e-07, '215': 1.5495455727432272e-06, '216': 3.587372532365407e-07, '217': 3.736846281299222e-07, '218': 3.138951001346868e-07, '219': 2.5410554371774197e-07, '22': 9.46667739754048e-08, '220': 2.5410554371774197e-07, '221': 1.6940370528573112e-07, '222': 1.0961415597421365e-07, '223': 1.6940370528573112e-07, '224': 1.1459661664048326e-07, '225': 6.477200287235974e-08, '226': 6.477200287235974e-08, '227': 6.477200287235974e-08, '228': 5.480707798710682e-08, '229': 4.484215665456759e-08, '23': 1.7438615884657338e-07, '230': 5.978954220609012e-08, '231': 3.985969243558429e-08, '232': 3.4877231769314676e-08, '233': 4.982461554448037e-09, '234': 1.494738555152253e-08, '235': 1.494738555152253e-08, '236': 4.982461554448037e-09, '237': 0.0, '238': 4.982461554448037e-09, '239': 0.0, '24': 1.1459661664048326e-07, '240': 9.964923108896073e-09, '241': 0.0, '242': 4.982461554448037e-09, '243': 4.982461554448037e-09, '244': 0.0, '245': 0.0, '246': 0.0, '247': 0.0, '248': 4.982461554448037e-09, '249': 0.0, '25': 1.544563161814949e-07, '250': 0.0, '251': 0.0, '252': 0.0, '253': 4.982461554448037e-09, '254': 4.982461554448037e-09, '255': 9.964923108896073e-09, '26': 1.7936862661827035e-07, '27': 2.5410554371774197e-07, '28': 3.6870216035822523e-07, '29': 3.8364956367331615e-07, '3': 0.0, '30': 3.6371969258652825e-07, '31': 3.9859693856669765e-07, '32': 4.3347418454686704e-07, '33': 6.228077040759672e-07, '34': 6.776148211429245e-07, '35': 7.025271315796999e-07, '36': 7.623166311532259e-07, '37': 8.719308084437216e-07, '38': 8.769132477937092e-07, '39': 9.66597553997417e-07, '4': 4.982461554448037e-09, '40': 1.0512994776945561e-06, '41': 1.2157206583651714e-06, '42': 1.4399314522961504e-06, '43': 1.4399314522961504e-06, '44': 1.7887037984110066e-06, '45': 2.0029497136420105e-06, '46': 3.70695147466904e-06, '47': 3.906250185536919e-06, '48': 3.5574776120483875e-06, '49': 4.1952325773308985e-06, '5': 4.982461554448037e-09, '50': 5.6102517191902734e-06, '51': 8.001833521120716e-06, '52': 1.1738679859263357e-05, '53': 1.2241908734722529e-05, '54': 1.499721020081779e-05, '55': 8.081054693320766e-05, '56': 0.00010066566028399393, '57': 5.782146763522178e-05, '58': 4.9067282816395164e-05, '59': 5.932617204962298e-05, '6': 1.494738555152253e-08, '60': 5.309809421305545e-05, '61': 5.4109535994939506e-05, '62': 8.377013000426814e-05, '63': 6.656070763710886e-05, '64': 6.162806676002219e-05, '65': 7.04569902154617e-05, '66': 6.87131323502399e-05, '67': 6.45577601972036e-05, '68': 6.667530396953225e-05, '69': 6.853874219814315e-05, '7': 2.989477110304506e-08, '70': 7.77812092565e-05, '71': 9.003308514365926e-05, '72': 9.659498755354434e-05, '73': 8.799525676295161e-05, '74': 8.238500595325604e-05, '75': 8.144829917000607e-05, '76': 8.389468712266535e-05, '77': 9.148796380031854e-05, '78': 9.624123049434274e-05, '79': 0.00010496054164832458, '8': 1.494738555152253e-08, '80': 0.00011071528570028022, '81': 0.00012170659465482458, '82': 0.00013531868171412498, '83': 0.00014155672397464514, '84': 0.000155562418513, '85': 0.0001895179011626169, '86': 0.0001875797170214355, '87': 0.00018620953778736293, '88': 0.0001958107459358871, '89': 0.0002262087509734556, '9': 1.9929846217792146e-08, '90': 0.00024018953263293952, '91': 0.0002340860228287056, '92': 0.0002454360655974597, '93': 0.0002618134312797338, '94': 0.00028089623083360493, '95': 0.0003357382083777338, '96': 0.0003852937661577016, '97': 0.0003892548265866935, '98': 0.00042850166209973395, '99': 0.0004571358731482178}
    # freq = {'0': 4.982461554448037e-09, '1': 9.964923108896073e-09, '10': 4.982461554448037e-09, '100': 0.00033395449281670153, '101': 0.0004288803320378065, '102': 0.0005232531693764031, '103': 0.0008211794192902744, '104': 0.0013579599326476455, '105': 0.0013399383751675487, '106': 0.0013590810121968389, '107': 0.0016525728860870004, '108': 0.0024033153895288706, '109': 0.003191969357430935, '11': 0.0, '110': 0.004579585045576096, '111': 0.005075474269688129, '112': 0.006221042014658451, '113': 0.007489760871976614, '114': 0.009421152994036674, '115': 0.012296357192099094, '116': 0.015540906228125095, '117': 0.01989356428384781, '118': 0.02706596814095974, '119': 0.03755755349993706, '12': 4.982461554448037e-09, '120': 0.04887263849377632, '121': 0.06292324513196945, '122': 0.08254551142454147, '123': 0.23509663343429565, '124': 0.09610658138990402, '125': 0.07018100470304489, '126': 0.05479946732521057, '127': 0.04346520081162453, '128': 0.03362031280994415, '129': 0.02529747225344181, '13': 4.982461554448037e-09, '130': 0.019402189180254936, '131': 0.015241081826388836, '132': 0.01119898445904255, '133': 0.008576461113989353, '134': 0.007071289233863354, '135': 0.004821583162993193, '136': 0.004032341297715902, '137': 0.0037714694626629353, '138': 0.0028741180431097746, '139': 0.001988415839150548, '14': 4.982461554448037e-09, '140': 0.001669249264523387, '141': 0.0013920898782089353, '142': 0.0007950513972900808, '143': 0.000585967383813113, '144': 0.00047056362382136285, '145': 0.0004130709858145565, '146': 0.0003426239709369838, '147': 0.00028968031983822584, '148': 0.0002510712365619838, '149': 0.00022817682474851608, '15': 9.964923108896073e-09, '150': 0.0001745107292663306, '151': 0.00014299665053840727, '152': 0.00012358497770037502, '153': 0.00011073521454818547, '154': 9.504045738140121e-05, '155': 8.325195085490122e-05, '156': 7.554906915174797e-05, '157': 6.111487891757861e-05, '158': 5.3810585086466745e-05, '159': 4.508629717747681e-05, '16': 9.964923108896073e-09, '160': 4.0766502934275195e-05, '161': 3.9127273339545354e-05, '162': 2.966557804029435e-05, '163': 2.118044540111441e-05, '164': 1.7314054275630042e-05, '165': 1.4020647540746722e-05, '166': 1.4214962902769912e-05, '167': 1.2700294973910786e-05, '168': 9.033203241415322e-06, '169': 6.761200438631931e-06, '17': 9.964923108896073e-09, '170': 6.2779017753200606e-06, '171': 5.480707841343246e-06, '172': 4.603794423019281e-06, '173': 4.1952325773308985e-06, '174': 3.756776095542591e-06, '175': 3.298389628980658e-06, '176': 3.128986008960055e-06, '177': 2.6656171030481346e-06, '178': 2.476283498253906e-06, '179': 2.117546273439075e-06, '18': 3.985969243558429e-08, '180': 1.7289141851506429e-06, '181': 1.3801419527226244e-06, '182': 1.0911591061812942e-06, '183': 1.2157206583651714e-06, '184': 1.1210538559680572e-06, '185': 9.715799933474045e-07, '186': 7.772640060466074e-07, '187': 7.374043207164505e-07, '188': 6.576849500561366e-07, '189': 5.829480187458103e-07, '19': 9.964923108896073e-09, '190': 5.630181476590224e-07, '191': 5.281409585222718e-07, '192': 4.5838646656193305e-07, '193': 4.085618741100916e-07, '194': 3.238600072563713e-07, '195': 4.1354431346007914e-07, '196': 2.790178541545174e-07, '197': 4.035794063383946e-07, '198': 3.6371969258652825e-07, '199': 2.7403538638282043e-07, '2': 4.982461554448037e-09, '20': 2.4912308660418603e-08, '200': 2.790178541545174e-07, '201': 1.893335479508096e-07, '202': 2.192283119484273e-07, '203': 1.544563161814949e-07, '204': 1.8435108017911261e-07, '205': 1.7936862661827035e-07, '206': 8.968431330913518e-08, '207': 1.893335479508096e-07, '208': 1.3452647351641644e-07, '209': 1.544563161814949e-07, '21': 1.9929846217792146e-08, '210': 1.245615379730225e-07, '211': 1.2954400574471947e-07, '212': 9.46667739754048e-08, '213': 7.473692420489897e-08, '214': 9.964923464167441e-08, '215': 6.477200287235974e-08, '216': 2.989477110304506e-08, '217': 6.477200287235974e-08, '218': 3.985969243558429e-08, '219': 5.978954220609012e-08, '22': 2.989477110304506e-08, '220': 3.4877231769314676e-08, '221': 2.989477110304506e-08, '222': 3.985969243558429e-08, '223': 3.985969243558429e-08, '224': 1.9929846217792146e-08, '225': 1.9929846217792146e-08, '226': 2.989477110304506e-08, '227': 2.989477110304506e-08, '228': 3.4877231769314676e-08, '229': 1.9929846217792146e-08, '23': 3.4877231769314676e-08, '230': 3.985969243558429e-08, '231': 1.494738555152253e-08, '232': 2.4912308660418603e-08, '233': 4.982461554448037e-09, '234': 9.964923108896073e-09, '235': 9.964923108896073e-09, '236': 4.982461554448037e-09, '237': 9.964923108896073e-09, '238': 4.982461554448037e-09, '239': 1.494738555152253e-08, '24': 3.4877231769314676e-08, '240': 1.494738555152253e-08, '241': 9.964923108896073e-09, '242': 4.982461554448037e-09, '243': 0.0, '244': 0.0, '245': 9.964923108896073e-09, '246': 0.0, '247': 0.0, '248': 0.0, '249': 0.0, '25': 4.484215665456759e-08, '250': 0.0, '251': 0.0, '252': 0.0, '253': 0.0, '254': 0.0, '255': 9.964923108896073e-09, '26': 3.4877231769314676e-08, '27': 4.9824617320837206e-08, '28': 7.473692420489897e-08, '29': 6.975446353862935e-08, '3': 4.982461554448037e-09, '30': 5.480707798710682e-08, '31': 9.964923464167441e-08, '32': 9.46667739754048e-08, '33': 1.0961415597421365e-07, '34': 7.971938487116859e-08, '35': 1.1459661664048326e-07, '36': 7.473692420489897e-08, '37': 9.46667739754048e-08, '38': 1.0463169530794403e-07, '39': 1.0463169530794403e-07, '4': 4.982461554448037e-09, '40': 1.6940370528573112e-07, '41': 1.893335479508096e-07, '42': 1.893335479508096e-07, '43': 1.5943876974233717e-07, '44': 1.8435108017911261e-07, '45': 2.2421077972012426e-07, '46': 2.192283119484273e-07, '47': 2.690529470328329e-07, '48': 2.341757010526635e-07, '49': 3.288424750280683e-07, '5': 0.0, '50': 2.989476968195959e-07, '51': 2.989476968195959e-07, '52': 4.534040272119455e-07, '53': 4.2849171677517006e-07, '54': 4.5838646656193305e-07, '55': 5.131935836288903e-07, '56': 5.032286480854964e-07, '57': 5.779655793958227e-07, '58': 6.676498855995305e-07, '59': 6.676498855995305e-07, '6': 1.494738555152253e-08, '60': 6.975446353862935e-07, '61': 8.021763164833828e-07, '62': 8.769132477937092e-07, '63': 1.0164221748709679e-06, '64': 1.0413344853077433e-06, '65': 1.3253347788122483e-06, '66': 1.444913891646138e-06, '67': 1.7189493064506678e-06, '68': 1.8484932979845325e-06, '69': 2.26203769670974e-06, '7': 0.0, '70': 2.381616695856792e-06, '71': 3.6670917324954644e-06, '72': 3.1937579478835687e-06, '73': 3.976004336436745e-06, '74': 4.693478786066407e-06, '75': 5.719866294384701e-06, '76': 7.37404343453818e-06, '77': 9.895168659568299e-06, '78': 1.1663943041639868e-05, '79': 1.365194475511089e-05, '8': 9.964923108896073e-09, '80': 1.6123245586641133e-05, '81': 1.7657845091889612e-05, '82': 2.264030626975e-05, '83': 2.3038903236738406e-05, '84': 2.443897392367944e-05, '85': 2.886340007535182e-05, '86': 3.438895146246068e-05, '87': 4.226622331771068e-05, '88': 4.557457577902824e-05, '89': 5.330237399903126e-05, '9': 9.964923108896073e-09, '90': 6.567382661160082e-05, '91': 8.342634100699797e-05, '92': 0.00010141800885321572, '93': 0.00010954938625218347, '94': 0.00011930504842894152, '95': 0.00014172612281981856, '96': 0.00016642418631818146, '97': 0.00019628906738944352, '98': 0.00023798229813110083, '99': 0.00027929688803851604}
    # h.compressNew(freqProj)
