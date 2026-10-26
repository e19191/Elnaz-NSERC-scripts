#!/usr/bin/env python3
"""KITCO3 descriptive publication figures 05b v1.1.0.

Replaces the presentation layout, not the estimator. All 71 pair phases; no
heatmaps. One three-page PDF per phase: RR/PLV/coverage, then each partner's
raw-versus-cleaned ECG. Offline window viewers contain ALL scored contexts.
Original Script05 v1.0.0 must be beside this file. See generated README.
"""
from __future__ import annotations
import argparse,base64,hashlib,html,importlib.util,json,math,os,platform,sys,zlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VERSION='1.1.0'
PROCESSING_SHA='acb6f6aecc1cae79cb8e8fec707d03e99ab80b8d3303cf755f6c24654628e4c0'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#CC79A7'
SOURCE_HASHES={
  "S001_P1_PID001_S2277_ATT_10685c19d340": "3165a333d9a46e6083e43f05b55ba8e3c2fc4072e4826455d68e46ca72d35744",
  "S001_P1_PID002_S2247_ATT_10685c19d340": "9120fc15a4974f514796cf38dbc2812fc98333aaf5808ccdc7dac13fadc162e8",
  "S001_P1_PID003_S2208_ATT_f1960ee4c304": "339cdf8ae92fa3bcd23ea03c044578f0b9d1dc81d727af971fef7a3652476ef1",
  "S001_P1_PID004_S2239_ATT_f1960ee4c304": "ff3c607a632190f497ccc35a47013176079af9c14626264234cba70dfe4533d9",
  "S001_P2_PID002_S2247_ATT_23dab7bde5f7": "7bd6530382ea4f3bc930c835bd757f9e138a1a9c1160b2246a7a41371ba3c3f9",
  "S001_P2_PID004_S2239_ATT_23dab7bde5f7": "19d8fb6d691af9e55de6fe5640a3e298b55c30d49bd24b9134456bb7f3ff8083",
  "S002_P1_PID005_S2208_ATT_337e4fb0e2d9": "cf35d4386adfae7f6f56af3fad54d30a88eb51e1ea21c540a55c83a7373b607a",
  "S002_P1_PID006_S2247_ATT_337e4fb0e2d9": "e43ce41035f31152b58451cc90e8ce53226cf25fed8aaa54d14d573aa75220b0",
  "S002_P1_PID007_S2238_ATT_6cde7343e048": "5a4e400588a4e520b659e7ffecb38e471a29caabcb9d33504fdfd636481654df",
  "S002_P1_PID008_S2239_ATT_6cde7343e048": "7b89375fd968978092b0a1d45ae1af6f80cae09d33edff117ba8115a73d7109e",
  "S002_P2_PID005_S2208_ATT_9fee298861fb": "14a86dd3ce530f22b431338b5b311b93b8b17f78168e27d9b9fcb162493bea9e",
  "S002_P2_PID006_S2247_ATT_c942e33ead12": "348874d20bfb9c58f1699fe9e6cdbc9a39a9b0315c83efa2c14afe4654f33d54",
  "S002_P2_PID007_S2238_ATT_9fee298861fb": "a5b1755aba0f903eae9212ab6581e78a8e3aeb49e1b308f74469e6a5b798385a",
  "S002_P2_PID008_S2239_ATT_c942e33ead12": "16ff0df16dbc7e6abe0914b1e64c037537a99a9c55c26b74a182ae576858e4b2",
  "S003_P1_PID009_S2208_ATT_8637ae0782eb": "4956ae6cbc6b108f33e35e9b79f79fde2e9366bce1fad9d1152e5771d4f0ac54",
  "S003_P1_PID010_S2247_ATT_8637ae0782eb": "7e62f28302c5c2a575776d84be0c0813b81d9f74673fd6f56bba8a4e9770a8b1",
  "S003_P1_PID011_S2238_ATT_6f6fe471002f": "f402a3562c350e17a5123867627bc36790b3cd89f3eaa0c99e6c606469717dd9",
  "S003_P1_PID012_S2239_ATT_6f6fe471002f": "59974572e5df8a6afb05c6bcf3ff99b172af7538305f1db1dc3828053b3dcd89",
  "S003_P2_PID009_S2208_ATT_1540bbe1c609": "e03e7329a31991fd3e424bdfce725d1f04055053df04b3ccf2a3460255692081",
  "S003_P2_PID010_S2247_ATT_05bd3ca33559": "acfcfbfd005b43e25c7e3b75238f08c111543df6f9a1037f95e4cb7062d012dc",
  "S003_P2_PID011_S2238_ATT_1540bbe1c609": "6382650aaa89eaf373284f0c19fffc2107add42f6916625cf77e8dfa94dbf943",
  "S003_P2_PID012_S2239_ATT_05bd3ca33559": "9b3be1ab13c41342ad1b1f03ce3c9b81554ff29b5bfb25977ca058d6918f5622",
  "S005_P1_PID017_S2239_ATT_46a162a424f7": "74d729e35c873ac3d360f92e3ccdb1df855160d24ef8c9fe3c2fb5a37036b62b",
  "S005_P1_PID018_S2277_ATT_46a162a424f7": "91e819dc593655427ff712a9a161c23ed074e4c5244bae76a4d5f7e8ec0a3e68",
  "S005_P1_PID019_S2247_ATT_e18aeb8b5cdf": "e09b0e8de8a39e96870cfcac3ace30c1c9069e297bdd34b55eb8c7dfff04ffab",
  "S005_P1_PID020_S2208_ATT_e18aeb8b5cdf": "a365b1243807a295051a4c46d3a07595b4868d75a646552125581c7c139a535a",
  "S005_P2_PID017_S2239_ATT_1049a1438b35": "955321047431a1786145dc05aafabd72f1c11430dde0881b147929ab868731b9",
  "S005_P2_PID018_S2277_ATT_51ccb15739da": "dd361874193b87df2c449f8a5cadb2a4810e343793b368bf440e54f67ed91d83",
  "S005_P2_PID019_S2247_ATT_1049a1438b35": "7c38c881d6edfa98cab51d29aa2ff622e5d29fae7f8bb10858a2cefafca4855c",
  "S005_P2_PID020_S2208_ATT_51ccb15739da": "5a7984967f9add5d4a55ed6f305dfddc2d5f3114a06716b5a7099dcd036970a5",
  "S006_P1_PID021_S2208_ATT_23664d49667b": "8d4bb2ac460b0716afac3e1a4b699523b047e7cc2660b1beb7def3816e41e39c",
  "S006_P1_PID022_S2239_ATT_23664d49667b": "92d3c7d4d761c389e768384303dabf324805a835b9e903b6c296e2643aae7aa3",
  "S006_P1_PID023_S2247_ATT_36600a2cee9d": "5d8560788b17be37adbc8715c8c0099f37cafa62025b72364170a6be6f19faef",
  "S006_P1_PID024_S2277_ATT_36600a2cee9d": "0589c16cb73e33fc600145f33cf9ba6fd968a55138641cd77a6e487908f0fae3",
  "S006_P2_PID021_S2208_ATT_c0ca538f2aaa": "c4f8ccf6204a0f0d7faa7b6141fdf893618341cf95c8aa343d216344866a5000",
  "S006_P2_PID023_S2247_ATT_c0ca538f2aaa": "2de04963ca7959ac4a20a8efb513f58860bfab50bd7234117c88aabd3bdd5169",
  "S007_P1_PID025_S2204_ATT_08881f11b2b5": "a16b3cb7c259c02cb7e7737c9b342cc3878d2ec89ffbceeb252d36e35d66a3cc",
  "S007_P1_PID026_S2208_ATT_08881f11b2b5": "8691182d0cc84f8bac70ed51b40ca9f818bf86267fcafa9c832b2ae762b3ac1d",
  "S007_P2_PID025_S2204_ATT_6522d5a84dcc": "574a9939c160e9be7f4a88d6c717bf7041d41f6d64c7da8fd9ef9e0f086a3886",
  "S007_P2_PID026_S2208_ATT_6522d5a84dcc": "243ae7a743aa9a13e908bdbfc4dc288aea9f5dd9026b69cc88e642cad39f3e58",
  "S008_P1_PID029_S2204_ATT_0e5a35007d2f": "6a32f59fc8370083f3f210a754f13891a37387670ee5b06f942a0ddf633a6e08",
  "S008_P1_PID030_S2208_ATT_0e5a35007d2f": "c01e225dab0321b174823e07bfd2f9c43bb091b27cfb3d9af2945cdfe5bc0b30",
  "S008_P2_PID029_S2204_ATT_e9597301d8a5": "c081f6b7b1fc6f6f08ff04a94a45ae184b77a8dc8b27522644389c6b53afb78e",
  "S008_P2_PID030_S2208_ATT_0584b9d3fca4": "b88e5128bb783cc96e67216c781b28e3007a6c4d74640163c4304d8d386e0efd",
  "S008_P2_PID031_S2247_ATT_e9597301d8a5": "8c436720971c43096bfca1b830de623c74feeb822954586755fd531439d221ab",
  "S008_P2_PID032_S2277_ATT_0584b9d3fca4": "8c1972a9aa8b2fb99a5c039e1afb7400d872b057aa32e2366fe0889a0e644b18",
  "S009_P2_PID035_S2247_ATT_80ad70e3f432": "88549bcfd5344acdee5beecb589f2a83ef40ec24df51ea1b5f591c44051e63a6",
  "S009_P2_PID036_S2277_ATT_80ad70e3f432": "97e7191e10363a7d531fa16a162fd72186e8820da74ea402b12d732634026ed3",
  "S010_P1_PID037_S2238_ATT_b5622508a273": "7ede8f1dab8eff23abd8258ca67a8849bf6a97393b2e0f07c28e459474f72192",
  "S010_P1_PID038_S2239_ATT_b5622508a273": "9b086ae522e3082cdeb4558e9e9098d1beb86e649d5aa4edc7f5cbcad3332410",
  "S010_P1_PID039_S2208_ATT_031f6d5d15e8": "3f1dcf44ad9e90cfb6b8f32a17fde9ce95888fb211984740253c3a11ff3e41d7",
  "S010_P1_PID040_S2168_ATT_031f6d5d15e8": "699a2643713845a914904f8925edb0254a178e5232bc04761f1f1a9298cf93d1",
  "S010_P2_PID038_S2239_ATT_425306b2790d": "452b019aac193f5abe6ebd91458c0dd5fe2611b7e84fadb1185bff2b17cf1341",
  "S010_P2_PID040_S2168_ATT_425306b2790d": "cc9be17a2f7e0e7a39a02b5be4aee37574cf5e8bec58b23b23f850851c2416a7",
  "S011_P2_PID041_S2168_ATT_51d4dc6a3bb2": "c3bbb6a37cb82541f0bbfb527c8ffd07166758d4f4ae6c72d64e86f0a2d67571",
  "S011_P2_PID043_S2238_ATT_51d4dc6a3bb2": "495ce48ae37fdc3c934fcad8dcba0f5e34b17c497d29e38d41713995e82599cf",
  "S012_P1_PID045_S2168_ATT_09e00babd8c0": "ac3727e890fe8e2522d0e93b3d4a8851558a4f8ecabc7a31f3a66487d76f3ed6",
  "S012_P1_PID046_S2208_ATT_09e00babd8c0": "322909a0fd33bd370a8143ba68e1a4c27f318a24f596052d5e318b79d629dd65",
  "S012_P1_PID047_S2238_ATT_63ed6257922f": "707f4da59b1d55ad3776ef9c2dd2e70f1dd5e768ac6635c1c7e78d507fb8e357",
  "S012_P1_PID048_S2239_ATT_63ed6257922f": "b719fc703f6a1fcfc9984339a65b365264ffc46152d46f339a84ba5492309f64",
  "S012_P2_PID045_S2168_ATT_f56b1bc05919": "7e4b6c21c19637d9b72365f3efb3498541e72d490825847571b7c2da3c27bc34",
  "S012_P2_PID046_S2208_ATT_4ffeec871fdb": "ff5f5ffb6f99a69bd2fc4ff9d999bfcef26ee638450fd93c024a2f0691f05f15",
  "S012_P2_PID047_S2238_ATT_f56b1bc05919": "1ec44663076497c2ebd50d04412e33fc3783835a19e60a0dc223ed33a4ebd669",
  "S012_P2_PID048_S2239_ATT_4ffeec871fdb": "406fbfdf909e4c36d30a964a36de37794609422f6e7df0f1fb14cb626440d460",
  "S013_P1_PID049_S2238_ATT_aa8b1977f28c": "b46104e3fae70d2bacbd2980b86c4725f16e84082e8153d711f9ecd24a0a94b0",
  "S013_P1_PID050_S2239_ATT_aa8b1977f28c": "19513cea5374367e54cb8d22a0345d4f7619c12f80e56ae425e9ffa8e042bb49",
  "S013_P2_PID049_S2238_ATT_0b3068d4edf1": "4775c2ecebf6bdc8216145ead98a07a28a6253f1815afa9339e7f6b23c43b20a",
  "S013_P2_PID051_S2247_ATT_0b3068d4edf1": "c2f1ef9e3e89ab8d50d13e65ad63c63c2e30775cc98ba2661de36ebdb3f4887c",
  "S014_P1_PID055_S2247_ATT_76aa92730452": "7e4ba0eb8cd2d9d4399842d7d1e9dae9c7f062792985e18553fba4807964b8fa",
  "S014_P1_PID056_S2277_ATT_76aa92730452": "5db379e6105414c7a1651bf5c4b80a08ac720f65f2a6861a3edaeda134e28a39",
  "S014_P2_PID053_S2238_ATT_f907d00e757d": "e9c2eb79eaf0d9e3408301cd53adfd02f9515c11cbfe79cc282cd90bcd37ad81",
  "S014_P2_PID054_S2239_ATT_2a4047235f1c": "e7f77b27b7d5afda216a18b9cd77a2fed0085105516e57c9bc59760605a0a634",
  "S014_P2_PID055_S2247_ATT_f907d00e757d": "a22fbe8cc96f95419d26437dd9f92c5ad612305223431798273cc5d23dedabd2",
  "S014_P2_PID056_S2277_ATT_2a4047235f1c": "9bdadf61369f253bf3de7d402c056d09504cccdfb741f71e9df485e2cfd20af7",
  "S015_P1_PID057_S2204_ATT_e7b719378940": "108dc95ef3c04a60575033426cd951f8b750019168a0ee47adda9fa7d33210c1",
  "S015_P1_PID058_S2205_ATT_e7b719378940": "6badad00c2a87c123d3edbf5598b55108be150c8eefbb4e923a3e7a578076799",
  "S015_P2_PID057_S2204_ATT_e2b3324dc559": "80af6d098307eccadbc763f9be5f203c76d7754c82007f42618ea33d365f0916",
  "S015_P2_PID058_S2205_ATT_759e1b6d0957": "9c6b588178d0d982af1acc3fb848ebf89f26b198166b201d548a7f8c9479cd7f",
  "S015_P2_PID059_S2208_ATT_e2b3324dc559": "eff74dc68d724e8a655a86bdddcc9197f93d85e095fe94b5f8da690712a45fe9",
  "S015_P2_PID060_S2168_ATT_759e1b6d0957": "379210819a771558a5960459aa456e6f182a66c9da9bef87c643019e51d11d80",
  "S016_P1_PID063_S2208_ATT_113365652f72": "753f217624f628270097112bb7764c7fa998cfe6ec07e455341500b12e3bf104",
  "S016_P1_PID064_S2168_ATT_113365652f72": "110181954c70876e8e729f34ffabd489868881ff738fe10ef52478bbe3750ad3",
  "S016_P2_PID062_S2205_ATT_d2f8db48cfba": "3b2ea64b230299343837e6640ee17b0ca28b42be4aebbae355b7e63e5fd319df",
  "S016_P2_PID064_S2168_ATT_d2f8db48cfba": "e3568e027af5d21cd1a348fd0e4b11a5deed6095a406923a81f5ddddef358efe",
  "S017_P1_PID065_S2204_ATT_bb64ce5cdc83": "a366bfcd06dbd12da7cdb6c8df1cdd229c94476958f1dcf667fbca050c0c251b",
  "S017_P1_PID066_S2205_ATT_bb64ce5cdc83": "5b21c37d426eb3625008a3379dc3363224b05c382a799b5f7cb57ceb23d67cb1",
  "S017_P2_PID065_S2204_ATT_077b04e017dd": "78208487de8ef95abd1b0b4f1fb3f3f1a94c9ee5d6f24b15de75340cec0bac66",
  "S017_P2_PID067_S2208_ATT_077b04e017dd": "f85171e90298cadf209d2428c9d9e1c00c3dc59d882fd9fe480c5de03b41ef80",
  "S018_P1_PID069_S2238_ATT_9f11c7fe5af2": "8edcf86a6b954f5f37c44d31efcf9112bb946c8a558f0503e5a597802b8168d9",
  "S018_P1_PID070_S2239_ATT_9f11c7fe5af2": "96da451cf5993e19681ef33fecdca33362e23187d79ba0fdf1fc9904d8133c64",
  "S018_P2_PID069_S2238_ATT_d7aa86997732": "01f3e521c02422e79cc3ad2789797ac326deaa17ba7558c4ee59ec66646a18c3",
  "S018_P2_PID070_S2239_ATT_e96311742147": "69f5cbee891158cf093ad715219fb121d13c21fa187b8575f0ba3481c581602a",
  "S018_P2_PID071_S2247_ATT_d7aa86997732": "4cb6428390749620f22b7208a270a2b31dfa31977e41e16a05169a884c4d190a",
  "S018_P2_PID072_S2277_ATT_e96311742147": "044f74ca2216fae2a0a6e9fb03436e873214b5885ac917c05e4f31e40c825595",
  "S019_P1_PID073_S2238_ATT_67a8fec89cb9": "b80c34fc3091456a141f9862e56743c7c58edb805792c715ba3f8497db8f640a",
  "S019_P1_PID074_S2239_ATT_67a8fec89cb9": "8cd51d7956070d510a7e6459997c8eadfd4e55612a670d86ec9ca136b5b5acc2",
  "S019_P2_PID074_S2239_ATT_e11e66f64875": "f3cad083502877430a4f58a0b346ee46c4a53faf99d9e2d909f6941bdb96fc7d",
  "S019_P2_PID076_S2277_ATT_e11e66f64875": "e40fb8816dbf6c6269b8b1a90ce600a90b223f8053ef984de1446a695cd87ea7",
  "S020_P1_PID077_S2238_ATT_b3bc17908559": "f360c2f2c2bd3089c55596a75b95b888d0fab8d7fbca4337305596c7ca3e3248",
  "S020_P1_PID078_S2239_ATT_b3bc17908559": "a63cd7a786611a9959bc4d8cb3bf7df94cda9078ff40b04e19161bf78a168f25",
  "S020_P1_PID079_S2247_ATT_676377fe7c13": "9180a319c457860ff70d86c43a936214660b3f188e2bb7273b35713f05110c96",
  "S020_P1_PID080_S2277_ATT_676377fe7c13": "360a199471db66b4594cee96ff54115e4fd4ba26d1c7b716a5327f7502ee305f",
  "S020_P2_PID077_S2238_ATT_c51271b79fab": "1ea64f49ed762cd227efe3ad2bb89f35fde7bff9c7e579b54732732b7c140c32",
  "S020_P2_PID078_S2239_ATT_de3938c16716": "092295e4ee4d76cff05660149c7142c82066a9d09afe92f7c318292339afa84e",
  "S020_P2_PID079_S2247_ATT_c51271b79fab": "09b82280fabf855fb136d1bbe21a1b8cbd2a97464c98df5470bfd059911243eb",
  "S020_P2_PID080_S2277_ATT_de3938c16716": "5d0eac5721ac74e5b031a5ae1e1920b2ac20eac3831707687b8cb40526437d73",
  "S021_P1_PID083_S2204_ATT_8bc8bdc97957": "a7d6d8d12994c01f5dd6116f7c66d99cfd380c5e33c7cd617adb2dac5ae14652",
  "S021_P1_PID084_S2277_ATT_8bc8bdc97957": "81f0d7e0aea7622672ce37c5e389196df75ae26ad74f2b1ba2b6f67165d20fa5",
  "S021_P2_PID081_S2238_ATT_9ddcef567559": "d8655a6c2d268fe7984e9e7dbc53d6355178b21662f41a613aaafb20a1a5c5bc",
  "S021_P2_PID083_S2204_ATT_9ddcef567559": "0aca3eb949ced33f20027305c69d492a5ae81d134ba64e88ced744d1bcca120e",
  "S022_P1_PID085_S2208_ATT_e6ef4e3a85df": "ccdb65733298e83f39c288777d6a62ed68c1b939bab76ffba3e0726a0a04d665",
  "S022_P1_PID086_S2204_ATT_e6ef4e3a85df": "757f955ea94d51bf8b07e87d9e016277e87836d37ce23ac801139c94b4fe3cf4",
  "S022_P1_PID087_S2238_ATT_e1a788bade4f": "167daa5172732256bb3fe05d72c5c42115c58886e3e886519e14a7cb8dcc13d8",
  "S022_P1_PID088_S2239_ATT_e1a788bade4f": "7bdd8b965ee1c1ad996810cea63b46754d85445f80a2dbbe4f1fabeeff1902aa",
  "S022_P2_PID085_S2208_ATT_7acf285f935f": "9163f43523e93476f2ab0c241ff3c5204270e309bc2ce8bae35e2c23831d228c",
  "S022_P2_PID086_S2277_ATT_11afef9e8030": "9328fa34f539a313dfc8bfa6aaeb779c8f5de27de4c1f687d7c63c6bc438246d",
  "S022_P2_PID087_S2238_ATT_7acf285f935f": "3341c1eaa62e233a50aafa14a6e0ebfca11c9f358b3ee028a172b2f9d94f91c5",
  "S022_P2_PID088_S2239_ATT_11afef9e8030": "e0ba29882931b07c86ecefa5e90d8df87561876f1d8a7810c457b86273f6fb4d",
  "S023_P1_PID091_S2239_ATT_d3e449e972ae": "6d4dcc03d8ade9079365fef5823bc490457c0d3eda6452610664febef796f536",
  "S023_P1_PID092_S2277_ATT_d3e449e972ae": "6de8c390909e02a417b4285b45dd9d16d873eed0d0f11aad28d3eceebda98461",
  "S023_P2_PID089_S2208_ATT_00327be6895f": "7371b088957e3c2c54ddde6d57098fe61ca5236cdf3d2ceb73f0cfccbc13aa5c",
  "S023_P2_PID091_S2239_ATT_00327be6895f": "ec4554c7e54e54aa4a9d71c29585763d08c26524fb0bd71218a052a02d314808",
  "S024_P1_PID093_S2208_ATT_d1a8fb6882de": "845d02711a35798a16a23292a6cbd0905a0a7e0c09727313624aa2361d0cc06a",
  "S024_P1_PID094_S2239_ATT_d1a8fb6882de": "463f946a61e94ec4688d5b2ca39a00fe4601bb39bd3e0e46ca9ea259fbbd2f92",
  "S024_P2_PID093_S2208_ATT_37f5c7bef0c7": "b1ff146f3d0ffd8644a0834d244737ab0c4dab573d9a3254d7171a8540eaf1b7",
  "S024_P2_PID094_S2239_ATT_37f5c7bef0c7": "c06e193d8b91578455e5e9d6361e14a8ac9782e29b678ce84a02dfad8ee7554f",
  "S025_P1_PID097_S2238_ATT_ea3a1c175351": "e3685b0f27475ab421f1b7b3f0d975daf76d059f64a612a6723f2bd73840bdaa",
  "S025_P1_PID098_S2239_ATT_ea3a1c175351": "9f93ae2074e774df84fcb629664122c85efb300c2da73e9459eb730f19c3fcd4",
  "S025_P1_PID099_S2168_ATT_3f1314915cc6": "bb2bcff3ae02d264317a8960f4e120611de4e9213bd831f7bcd31144762cc779",
  "S025_P1_PID100_S2277_ATT_3f1314915cc6": "6bd564e2a6dba8a084c77c7a0abaa7eac564e254fbb477a9cfd71d5d9f20893a",
  "S025_P2_PID097_S2238_ATT_29027f11439d": "40b069483942c322a26d5ab5937f0948f6db3b87f4b7b5a4104ad8fce330d3c2",
  "S025_P2_PID098_S2204_ATT_31c81b7ae9bf": "1c02fe8d020f04713abe79cd862f2f0056a748bf6a340c09a2cbe3039d061ed4",
  "S025_P2_PID099_S2168_ATT_29027f11439d": "55829dc12d1e58db5b508c1e02cdcd74d9d3c0f272b5e47ab75565c8c3b47e5a",
  "S025_P2_PID100_S2277_ATT_31c81b7ae9bf": "1c5760c06e55828143ab3e8a75f8caa70b27af6685f10291cd92e484848fbf11",
  "S026_P1_PID101_S2204_ATT_818c618ce1f0": "da8d4fd08a1a47c14bcf8564033095100900a822c15ad7086dc0133618c91bd8",
  "S026_P1_PID102_S2168_ATT_818c618ce1f0": "77f0ca22bd97884d04ed08372fc777f801deed0ce3f27bb5b8e1f9a494d0b7b9",
  "S026_P1_PID103_S2238_ATT_0c3ef6cc9639": "650e0be4b8a3e70cd351addfaf0116af0d09f5112511a5294c9a998a43b977eb",
  "S026_P1_PID104_S2277_ATT_0c3ef6cc9639": "2c474a8c082158beb4acbd04e91b6e2d12ce8ecb2abdf57a13f4ecba820320f2",
  "S026_P2_PID101_S2204_ATT_7409a4ff3286": "3bcf1305b804fb13df900fc6d0d640c08d815524b1c3a764513e0fa3528eb5a4",
  "S026_P2_PID102_S2168_ATT_2734c09e9f5b": "ae48114a23117f8575618fea561d74a8b257d1c68d6e509371cd898d78089af8",
  "S026_P2_PID103_S2238_ATT_7409a4ff3286": "6a601c1cdbcb48f85dc7b436c9eaeb4f760b66b66a0964f5750a80bfe1c30ad3",
  "S026_P2_PID104_S2277_ATT_2734c09e9f5b": "d6d40f8b14aaf808d750bdca10d97b88d15b32594c740f99ffce950219bd16df"
}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def publish(tmp,path):
    # Close, validate and sync each complete artifact before replacing its name.
    with tmp.open('rb+') as f:
        f.seek(max(0,tmp.stat().st_size-32));tail=f.read()
        if path.suffix=='.pdf':require(tail.rstrip().endswith(b'%%EOF'),'Incomplete PDF export: '+path.name)
        if path.suffix=='.png':require(tail.endswith(b'IEND\xaeB`\x82'),'Incomplete PNG export: '+path.name)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def atomic_text(path,text):
    tmp=path.with_name(path.name+'.writing');tmp.write_text(text,encoding='utf-8');publish(tmp,path)

def savefig(fig,path):
    tmp=path.with_name(path.stem+'.writing'+path.suffix)
    fig.savefig(tmp,dpi=300,facecolor='white');publish(tmp,path)

@contextmanager
def book(path):
    tmp=path.with_name(path.stem+'.writing.pdf')
    with PdfPages(tmp) as b:yield b
    publish(tmp,path)

def spans(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]+1e-8:out[-1][1]=max(out[-1][1],float(b))
        else:out.append([float(a),float(b)])
    return out

def window_runs(w):
    for _,g in w.sort_values(['paired_segment_id','core_start_s']).groupby('paired_segment_id'):
        for ix in np.split(np.arange(len(g)),np.flatnonzero(np.diff(g.core_start_s)>1.0000001)+1):
            if len(ix):yield g.iloc[ix]

def styling():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})

def tidy(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)

def main_figure(r,s,w,g,segments,zoom):
    fig,axes=plt.subplots(5,1,figsize=(10.5,12.3),gridspec_kw={'height_ratios':[1.15,1,1,.45,.65]})
    fig.subplots_adjust(left=.13,right=.95,top=.88,bottom=.23,hspace=.85)
    fig.suptitle(f'{r.pair_id} | RR intervals and phase consistency',x=.13,ha='left',y=.97,fontsize=14,fontweight='bold')
    fig.text(.13,.932,'RR interval (IBI) = time between ECG R-peaks. Phase is estimated from 0.04-0.15 Hz RR fluctuations.',fontsize=9)
    for ax in axes[:4]:tidy(ax);ax.set_xlim(0,r.shared_cooking_s/60)
    if g is not None:
        for k,part in enumerate(g.groupby('paired_segment_id',sort=True)):
            _,d=part
            for j,c in [(1,BLUE),(2,ORANGE)]:
                axes[0].plot(d.cooking_s/60,d[f'ibi_{j}_s']*1000,color=c,lw=.65,
                             label=f'PID {r.pair_id.split("_")[-3+j]}' if k==0 else None)
        axes[0].legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
    else:axes[0].text(.5,.5,'No jointly supported RR-interval grid',transform=axes[0].transAxes,ha='center');axes[0].set_yticks([])
    axes[0].set_ylabel('RR interval (IBI)\n(ms)');axes[0].set_title('A  Paired RR intervals on the exported time axis',loc='left')
    for ax in axes[1:3]:
        for q in window_runs(w):ax.plot((q.core_start_s+100)/60,q.PLV,color=BLUE,lw=1,marker='.' if len(q)==1 else None)
        ax.set_ylabel('PLV')
        if len(w):ax.axhline(s.mean_local_PLV,color='#555555',ls='--',lw=.8)
    axes[1].set_ylim(0,1);axes[1].set_yticks([0,.5,1]);axes[1].set_title('B  PLV: full 0-1 scale',loc='left')
    axes[2].set_ylim(0,zoom);axes[2].set_title(f'C  Same PLV values: enlarged scale 0-{zoom:g}, shared across all pairs',loc='left')
    if len(w):
        axes[1].text(.99,.92,f'Mean local PLV: {s.mean_local_PLV:.3f}\n{len(w)} dependent, overlapping windows',
                     transform=axes[1].transAxes,ha='right',va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    else:
        reason='No paired RR grid' if g is None else 'No continuous segment supports the required 280 s context'
        for ax in axes[1:3]:ax.text(.5,.5,reason,transform=ax.transAxes,ha='center',fontsize=9)
    for sr in segments.itertuples(index=False):
        axes[3].broken_barh([(sr.first_grid_s/60,(min(sr.last_grid_s+.25,r.shared_cooking_s)-sr.first_grid_s)/60)],(.6,.22),facecolors='#AAAAAA')
    for a,b in spans([(x,x+200) for x in w.core_start_s]):axes[3].broken_barh([(a/60,(b-a)/60)],(.14,.22),facecolors=BLUE)
    axes[3].set_ylim(0,1);axes[3].set_yticks([.25,.71],['Scored core','Paired RR grid'],fontsize=8)
    axes[3].spines[['left','bottom']].set_visible(False);axes[3].set_xlabel('Cooking time (min)')
    paired=sum(b-a for a,b in spans([(x.first_grid_s,min(x.last_grid_s+.25,r.shared_cooking_s)) for x in segments.itertuples(index=False)]))
    axes[3].set_title(f'D  Coverage: paired grid {paired/r.shared_cooking_s:.1%}; scored-core union {s.core_union_s/r.shared_cooking_s:.1%} of shared cooking',loc='left',fontsize=9)
    ax=axes[4];tidy(ax);ax.set_xlim(-45,245);ax.set_ylim(0,1)
    ax.broken_barh([(-40,40),(200,40)],(.45,.3),facecolors='#CCCCCC')
    ax.broken_barh([(0,200)],(.45,.3),facecolors=BLUE)
    ax.text(-20,.25,'40 s context',ha='center',fontsize=8);ax.text(100,.25,'200 s scoring core',ha='center',fontsize=9);ax.text(220,.25,'40 s context',ha='center',fontsize=8)
    ax.annotate('PLV plotted at core midpoint',xy=(100,.77),xytext=(100,.98),ha='center',fontsize=9,arrowprops={'arrowstyle':'->','lw':.8})
    ax.set_xticks([-40,0,100,200,240]);ax.set_yticks([]);ax.spines[['left','bottom']].set_visible(False)
    ax.set_xlabel('Seconds relative to scoring-core start');ax.set_title('E  Window definition (schematic)',loc='left',pad=8)
    fig.text(.13,.116,'Each PLV uses a 200 s core plus 40 s real context per side. Starts advance 1 s within supported runs.\n'
             'Adjacent windows share 199 s of scoring data. Dashed line is the mean, not a significance threshold.\n'
             'Coverage is the union of scored periods, not window count multiplied by 200 s. Gaps are not bridged.',fontsize=8.5,linespacing=1.55)
    fig.text(.13,.047,'Descriptive working specification. Physical clock offset/drift is unverified; final inclusion is not frozen.\n'
             'No condition comparison, surrogate significance test, or independent window-based N is reported.\n'
             'Following pages: raw versus reconstructed cleaned ECG for each partner. Window detail: matching HTML viewer.',fontsize=8.2,linespacing=1.5,color='#444444')
    return fig

def phase_payload(proc,r,w,g,segments):
    pieces=[];order=[];maximum=0.
    for sr in segments.itertuples(index=False):
        sw=w.loc[w.paired_segment_id.eq(sr.paired_segment_id)].sort_values('core_start_s')
        starts=proc.window_starts(sr,r.shared_cooking_s,proc.family('working_200s_40s'))
        require(np.array_equal(starts,sw.core_start_s.to_numpy()),'Scored context support changed: '+r.pair_id)
        if not len(sw):continue
        signal=g.loc[g.paired_segment_id.eq(sr.paired_segment_id)].sort_values('cooking_s')
        for first in range(0,len(sw),64):
            q=sw.iloc[first:first+64];x,y=proc.contexts(signal,q.core_start_s.to_numpy(),proc.family('working_200s_40s'))
            a,b=proc.analytic(x),proc.analytic(y);score=proc.plv(a[:,160:960],b[:,160:960])
            err=float(np.max(abs(score-q.PLV.to_numpy())));maximum=max(maximum,err)
            require(err<=1e-8,'Context PLV differs from Script05')
            pieces.append(np.stack([a.real,b.real,np.angle(a*np.conj(b))],axis=1).astype('<f4'))
            order.extend([[float(v.core_start_s),int(v.paired_segment_id),float(v.PLV)] for v in q.itertuples(index=False)])
    if not pieces:return '',[],maximum
    matrix=np.concatenate(pieces)
    # Verify the display-precision phase itself reproduces all window PLVs.
    display=np.abs(np.mean(np.exp(1j*matrix[:,2,160:960].astype(float)),axis=1))
    require(np.max(abs(display-np.array(order)[:,2]))<1e-7,'Display precision changed phase consistency')
    return base64.b64encode(zlib.compress(matrix.tobytes(),6)).decode('ascii'),order,maximum

VIEWER=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ | RR interval and PLV viewer</title><style>
body{font:16px system-ui,sans-serif;line-height:1.5;margin:24px auto;padding:0 20px;max-width:1150px;color:#222;background:#fff}
h1{font-size:24px}h2{font-size:18px;margin-bottom:4px}label{display:inline-block;margin-right:20px}button,select,input{font:inherit}input[type=range]{width:min(650px,85vw)}
svg{width:100%;height:auto;overflow:visible}svg text{font:13px system-ui;fill:#222}.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}.note{font-size:14px;color:#444}
@media(max-width:750px){.grid{display:block}body{padding:0 12px;margin:15px auto}}
</style><h1>__TITLE__: RR interval (IBI) and PLV</h1>
<p>Every available scoring window is selectable. No window is selected because its PLV is high or low. <a href="../pairs/__TITLE__.pdf">PDF: RR, PLV and both partners' ECG comparisons</a></p>
<p id="state" role="status" aria-live="polite">Loading locally embedded data...</p>
<div id="controls" hidden><label for="win">Scored window <select id="win"></select></label><button id="prev">Previous</button> <button id="next">Next</button><br>
<input id="slider" type="range" min="0" max="0" value="0" aria-label="Scored window index"></div>
<h2>PLV through cooking: full scale and shared enlarged scale</h2><svg id="plv" role="img" aria-label="PLV with selected window marked"></svg>
<h2>Selected window: both partners' RR intervals</h2><svg id="rr" role="img" aria-label="RR interval in milliseconds with context"></svg>
<h2>Selected window: filtered RR fluctuations</h2><svg id="filtered" role="img" aria-label="Context standardized filtered RR fluctuations"></svg>
<div class="grid"><section><h2>Phase difference within scoring core</h2><svg id="phase" role="img" aria-label="Wrapped phase difference in radians"></svg></section>
<section><h2>Circular phase-difference distribution</h2><svg id="circle" role="img" aria-label="Fraction of samples in each 20 degree angular bin"></svg></section></div>
<p class="note">Blue: partner 1; orange: partner 2. Grey shading marks the 40 s contexts outside the 200 s scoring core. RR intervals are shown in ms; IBI is another label for these ECG-derived RR intervals. Filtered RR values use the original context standardization. Phase wraps at +/-pi; lines break at wraps. The circular bars show the fraction of the 800 core samples in each 20-degree bin, not independent observations. The labelled radial scale adapts to each selected window; compare the numerical PLV to assess phase consistency across windows.</p>
<p class="note">200 s working cores, 40 s context per side, 1 s start step, 0.04-0.15 Hz band. Contexts are processed separately exactly as in Script05. Display arrays are float32; PLV labels use sealed full-precision results. No temporal downsampling. Overlapping windows are dependent. Physical sensor-clock synchronization remains unverified. No significance test or condition effect is inferred.</p>
<script>
const META=__META__,PACK='__PACK__',RRPACK='__RRPACK__';
const NS='http://www.w3.org/2000/svg',blue='#0072B2',orange='#D55E00';
function el(tag,attrs={},parent){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(parent)parent.appendChild(n);return n;}
function text(svg,x,y,t,anchor='middle'){const n=el('text',{x,y,'text-anchor':anchor},svg);n.textContent=t;return n;}
function frame(id,xmin,xmax,ymin,ymax,xlabel,ylabel,height=245){const s=document.getElementById(id);s.replaceChildren();const W=Math.max(360,s.clientWidth),H=height,L=80,R=25,T=20,B=55;s.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const x=t=>L+(t-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
 el('path',{d:`M${L} ${T}V${H-B}H${W-R}`,fill:'none',stroke:'#444'},s);
 for(let i=0;i<=4;i++){let a=xmin+(xmax-xmin)*i/4,b=ymin+(ymax-ymin)*i/4;text(s,x(a),H-B+20,+a.toFixed(2));text(s,L-10,y(b)+4,+b.toFixed(2),'end');}
 text(s,(L+W-R)/2,H-8,xlabel);const n=text(s,16,(T+H-B)/2,ylabel);n.setAttribute('transform',`rotate(-90 16 ${(T+H-B)/2})`);return {s,x,y,W,H,L,R,T,B};}
function line(f,xs,ys,color,wrap=false){let d='',last=null;for(let i=0;i<xs.length;i++){if(!Number.isFinite(ys[i])){last=null;continue;}let move=last===null||(wrap&&Math.abs(ys[i]-last)>Math.PI);d+=(move?'M':'L')+f.x(xs[i])+','+f.y(ys[i]);last=ys[i];}el('path',{d,fill:'none',stroke:color,'stroke-width':1.2},f.s);}
function contexts(f){for(const [a,b]of [[-40,0],[200,240]])el('rect',{x:f.x(a),y:f.T,width:f.x(b)-f.x(a),height:f.H-f.B-f.T,fill:'#eeeeee'},f.s);}
function bounds(a,b){const v=[...a,...b],lo=Math.min(...v),hi=Math.max(...v),pad=Math.max((hi-lo)*.08,.01);return [lo-pad,hi+pad];}
async function unpack(encoded){const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));return new Float32Array(await new Response(stream).arrayBuffer());}
let matrix,rr,selected=0;
function draw(){const row=META.windows[selected],start=row[0],segment=row[1],plv=row[2],N=1120,off=selected*3*N;
 const t=Array.from({length:N},(_,i)=>-40+i/4),a=Array.from(matrix.subarray(off,off+N)),b=Array.from(matrix.subarray(off+N,off+2*N)),d=Array.from(matrix.subarray(off+2*N,off+3*N));
 document.getElementById('state').textContent=`${META.pair} | window ${selected+1}/${META.windows.length} | core ${start.toFixed(2)}-${(start+200).toFixed(2)} s | midpoint ${((start+100)/60).toFixed(2)} min | PLV ${plv.toFixed(4)} | paired segment ${segment}`;
 for(const id of ['win','slider'])document.getElementById(id).value=selected;
 const p=frame('plv',0,META.duration/60,0,1,'Cooking time (min)','PLV',230);
 let run=[];function flush(){if(run.length)line(p,run.map(q=>(q[0]+100)/60),run.map(q=>q[2]),blue);run=[];}
 for(const q of META.windows){if(run.length&&(q[1]!==run.at(-1)[1]||q[0]-run.at(-1)[0]>1.0001))flush();run.push(q);}flush();
 el('circle',{cx:p.x((start+100)/60),cy:p.y(plv),r:4,fill:orange},p.s);
 // One inset uses the same global enlarged scale for every pair.
 const sw=(p.W-p.L-p.R)*.40,sx=p.W-p.R-sw,sy=p.T+6,sh=65;
 el('rect',{x:sx,y:sy,width:sw,height:sh,fill:'#fff',stroke:'#aaa'},p.s);
 const z={s:p.s,x:v=>sx+v/(META.duration/60)*sw,y:v=>sy+sh-v/META.zoom*sh};run=[];
 function zflush(){if(run.length)line(z,run.map(q=>(q[0]+100)/60),run.map(q=>q[2]),blue);run=[];}
 for(const q of META.windows){if(run.length&&(q[1]!==run.at(-1)[1]||q[0]-run.at(-1)[0]>1.0001))zflush();run.push(q);}zflush();text(p.s,sx+sw/2,sy+sh+17,`Enlarged PLV: 0-${META.zoom}`);
 const rt=[],ra=[],rb=[];for(let i=0;i<rr.length;i+=4){if(rr[i+3]===segment&&rr[i]>=start-40-1e-5&&rr[i]<start+240-1e-5){rt.push(rr[i]-start);ra.push(rr[i+1]);rb.push(rr[i+2]);}}
 const lim=bounds(ra,rb),f=frame('rr',-40,240,...lim,'Seconds relative to core start','RR interval (IBI), ms');contexts(f);line(f,rt,ra,blue);line(f,rt,rb,orange);
 const amp=Math.max(...a.map(Math.abs),...b.map(Math.abs),.01)*1.05,g=frame('filtered',-40,240,-amp,amp,'Seconds relative to core start','Filtered RR (context units)');contexts(g);line(g,t,a,blue);line(g,t,b,orange);
 const ph=d.slice(160,960),pt=t.slice(160,960),h=frame('phase',0,200,-Math.PI,Math.PI,'Seconds within scoring core','Phase difference (rad)',270);line(h,pt,ph,blue,true);
 const svg=document.getElementById('circle');svg.replaceChildren();const W=Math.max(360,svg.clientWidth),H=300;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const cx=W/2,cy=H/2,R=100,bins=Array(18).fill(0);
 for(const v of ph)bins[Math.min(17,Math.floor((v+Math.PI)/(2*Math.PI)*18))]++;
 const ymax=Math.max(.1,Math.ceil(Math.max(...bins)/800*10)/10);
 for(let k=1;k<=2;k++){const rad=R*k/2;el('circle',{cx,cy,r:rad,fill:'none',stroke:'#ccc'},svg);text(svg,cx+5,cy-rad+13,`${(ymax*k/2).toFixed(2)}`,'start');}
 for(let i=0;i<18;i++){const a0=-Math.PI+i*2*Math.PI/18,a1=a0+2*Math.PI/18-.02,rad=R*(bins[i]/800)/ymax,x0=cx+rad*Math.cos(a0),y0=cy-rad*Math.sin(a0),x1=cx+rad*Math.cos(a1),y1=cy-rad*Math.sin(a1);el('path',{d:`M${cx} ${cy}L${x0} ${y0}A${rad} ${rad} 0 0 0 ${x1} ${y1}Z`,fill:blue,opacity:.8},svg);}
 text(svg,cx+R+15,cy+5,'0');text(svg,cx-R-22,cy+5,'+/-pi');text(svg,cx,cy-R-12,'pi/2');text(svg,cx,cy+R+22,'-pi/2');text(svg,cx,H-4,'Radius: fraction of samples per 20-degree bin');
 document.getElementById('prev').disabled=selected===0;document.getElementById('next').disabled=selected===META.windows.length-1;
}
(async()=>{try{if(!META.windows.length){document.getElementById('state').textContent=META.reason;for(const id of ['plv','rr','filtered','phase','circle'])document.getElementById(id).remove();return;}
 matrix=await unpack(PACK);rr=await unpack(RRPACK);if(matrix.length!==META.windows.length*3*1120)throw Error('Context array length mismatch');
 const select=document.getElementById('win');for(let i=0;i<META.windows.length;i++){const o=document.createElement('option');o.value=i;o.textContent=`${i+1}: midpoint ${((META.windows[i][0]+100)/60).toFixed(2)} min`;select.appendChild(o);}
 document.getElementById('slider').max=META.windows.length-1;document.getElementById('controls').hidden=false;
 select.onchange=e=>{selected=+e.target.value;draw();};document.getElementById('slider').oninput=e=>{selected=+e.target.value;draw();};
 document.getElementById('prev').onclick=()=>{selected--;draw();};document.getElementById('next').onclick=()=>{selected++;draw();};
 new ResizeObserver(()=>draw()).observe(document.getElementById('plv'));draw();document.documentElement.dataset.viewerReady='true';
 }catch(e){document.getElementById('state').textContent='Viewer could not load: '+e.message;document.getElementById('state').setAttribute('role','alert');}})();
</script></html>'''

def viewer(out,r,packed,order,g,zoom):
    reason='No jointly supported RR grid' if g is None else 'No continuous segment supports a 200 s core plus 40 s context per side'
    meta={'pair':r.pair_id,'duration':r.shared_cooking_s,'zoom':zoom,'windows':order,'reason':reason}
    rr='' if g is None else base64.b64encode(zlib.compress(np.column_stack([g.cooking_s,g.ibi_1_s*1000,g.ibi_2_s*1000,g.paired_segment_id]).astype('<f4').tobytes(),6)).decode('ascii')
    text=VIEWER.replace('__TITLE__',html.escape(r.pair_id)).replace('__META__',json.dumps(meta,separators=(',',':'))).replace('__PACK__',packed).replace('__RRPACK__',rr)
    atomic_text(out,text)

def display_indices(t,max_bins=1000):
    if len(t)<=2*max_bins:return np.arange(len(t))
    return None

def minmax_indices(x,bins=1000):
    if len(x)<=2*bins:return np.arange(len(x))
    edges=np.linspace(0,len(x),bins+1).astype(int);ix=[]
    for a,b in zip(edges[:-1],edges[1:]):
        if b>a:ix.extend([a+int(np.argmin(x[a:b])),a+int(np.argmax(x[a:b]))])
    return np.unique(ix)

def ecg_figure(uid,pair,source,peaks,withheld,nk,inputs):
    meta_path=inputs['src2']/'records'/uid/'02_result.json'
    require(meta_path.is_file(),'Missing Script02 recording metadata: '+str(meta_path))
    meta=json.loads(meta_path.read_text());require(meta['recording_uid']==uid,'Recording identity mismatch')
    raw=(inputs['raw']/meta['ECG_path']).resolve();require(raw.is_relative_to(inputs['raw']),'Unsafe raw ECG path')
    require(raw.is_file(),'Missing original raw ECG: '+str(raw))
    require(sha(raw)==SOURCE_HASHES[uid],'Original raw ECG hash differs: '+uid)
    inputs['hashes'][str(meta_path)]=sha(meta_path);inputs['hashes'][str(raw)]=SOURCE_HASHES[uid]
    frame=pd.read_csv(raw,skiprows=int(meta['header_line_number'])-1)
    t=frame.Timestamp.to_numpy(float);x=frame.Sample.to_numpy(float)
    require(np.isfinite(t).all() and np.isfinite(x).all() and len(t)==meta['valid_rows'],'Raw rows differ from the reviewed finite source')
    clean=np.full(len(x),np.nan);segments=[]
    for seg in meta['segments']:
        a,b=int(seg['start_valid_index']),int(seg['stop_valid_index_exclusive'])
        require(abs(t[a]-seg['start_timestamp'])<2e-6 and abs(t[b-1]-seg['stop_timestamp'])<2e-6,'Original processing segment mismatch')
        segments.append((a,b))
        if seg['status']=='PROCESSED':
            if meta['powerline_filter_applied']:
                clean[a:b]=nk.ecg_clean(x[a:b],sampling_rate=meta['sampling_rate_hz_used'],method='neurokit',powerline=meta['powerline_hz_requested'])
            else:
                clean[a:b]=nk.signal_filter(x[a:b],sampling_rate=meta['sampling_rate_hz_used'],lowcut=.5,method='butterworth',order=5)
    start=source.cooking_start_timestamp;duration=source.cooking_stop_timestamp-start;ct=t-start
    p=peaks.loc[peaks.recording_uid.eq(uid)].copy();idx=p.source_sample_index.to_numpy(int)
    require(((idx>=0)&(idx<len(t))).all(),'Reviewed peak outside raw ECG')
    if len(idx):require(np.max(abs(t[idx]-p.unix_s.to_numpy()))<2e-6 and np.array_equal(x[idx],p.raw_sample_value.to_numpy()),'Final reviewed peaks do not match raw source samples')
    require(p.trusted.isin([True,False]).all(),'Ambiguous peak trust values')
    exclusions=withheld.loc[withheld.recording_uid.eq(uid)]
    fig,axes=plt.subplots(4,2,figsize=(10.5,12.3));fig.subplots_adjust(left=.12,right=.95,top=.85,bottom=.23,hspace=.6,wspace=.22)
    fig.suptitle(f'{pair} | PID {int(source.PID):03d}: ECG comparison',x=.12,ha='left',y=.97,fontsize=14,fontweight='bold')
    fig.text(.12,.93,f'Raw ECG and reconstructed Script02 cleaning; final Script03 peak markers. Design rate: {meta["sampling_rate_hz_used"]} Hz.',fontsize=9)
    fig.text(.12,.9,'Rows: full cooking overview, then fixed 10 s intervals centred at 25%, 50% and 75% of cooking.',fontsize=9)
    intervals=[(0,duration)]+[(max(0,q*duration-5),min(duration,q*duration+5)) for q in [.25,.5,.75]]
    for row,(lo,hi) in enumerate(intervals):
        visible=np.flatnonzero((ct>=lo)&(ct<hi));finite=clean[visible][np.isfinite(clean[visible])]
        both=np.r_[x[visible],finite] if len(visible) else np.array([-1.,1.]);low=float(both.min());high=float(both.max());pad=max((high-low)*.05,1.)
        for col,values in enumerate([x,clean]):
            ax=axes[row,col];tidy(ax);ax.set_xlim(lo,hi);ax.set_ylim(low-pad,high+pad)
            for a,b in segments:
                si=visible[(visible>=a)&(visible<b)];si=si[np.isfinite(values[si])]
                if len(si):
                    if row==0:si=si[minmax_indices(values[si])]
                    ax.plot(ct[si],values[si],color=BLUE if col==0 else ORANGE,lw=.65)
            for e in exclusions.itertuples(index=False):
                left=max(lo,e.start_cooking_s);right=min(hi,e.stop_cooking_s)
                if right>left:ax.axvspan(left,right,color='#AAAAAA',alpha=.3,lw=0)
            if row:
                for trust,color,marker in [(True,GREEN,'o'),(False,PURPLE,'x')]:
                    selected=p.trusted.eq(trust).to_numpy()&(ct[idx]>=lo)&(ct[idx]<hi)&np.isfinite(values[idx])
                    ax.scatter(ct[idx[selected]],values[idx[selected]],s=20,c=color,marker=marker,zorder=3)
            ax.set_xlabel('Cooking time (s)')
            if col==0:ax.set_ylabel('ECG\n(exported units)')
            ax.set_title(('Raw ECG' if col==0 else 'Reconstructed cleaned ECG')+(f' | {lo:.1f}-{hi:.1f} s' if row else ' | full cooking'),loc='left',fontsize=9)
    fig.text(.12,.105,'Green circles: final trusted peaks. Purple crosses: retained untrusted peaks. Grey: withheld spans.\n'
             'Raw and cleaned columns share the same amplitude limits within each row. Overview uses min/max-preserving\n'
             'display reduction; 10 s waveform rows retain every sample. Fixed intervals are not chosen for signal quality.',fontsize=8.2,linespacing=1.5)
    fig.text(.12,.038,'Cleaning arrays were not stored by Script02; traces are reconstructed using its recorded settings and NeuroKit2 0.2.13.\n'
             'No R-peaks are redetected or changed. Source hashes and every displayed reviewed peak anchor are checked.\n'
             'Exported amplitude units are not calibrated microvolts. Filter design rate is not physical clock verification.',fontsize=8,linespacing=1.5,color='#444444')
    return fig

def overview(reg,summaries,first,last):
    fig,axes=plt.subplots(1,2,figsize=(10.5,10),sharey=True);fig.subplots_adjust(left=.25,right=.95,top=.87,bottom=.20,wspace=.18)
    for i,r in enumerate(reg.iloc[first:last].itertuples(index=False)):
        s=summaries.loc[r.pair_id]
        if s.windows:axes[0].plot(s.mean_local_PLV,i,'o',color=BLUE,ms=5)
        else:axes[0].text(.02,i,'Unavailable',va='center',fontsize=8,color='#666666')
        axes[1].barh(i,r.shared_cooking_s/60,height=.6,color='#DDDDDD');axes[1].barh(i,s.core_union_s/60,height=.32,color=BLUE)
    axes[0].set_yticks(range(last-first),reg.iloc[first:last].pair_id,fontsize=8);axes[0].invert_yaxis();axes[0].set_xlim(0,1)
    axes[1].set_xlim(0,reg.shared_cooking_s.max()/60*1.04)
    axes[0].set_xlabel('Mean local PLV');axes[1].set_xlabel('Duration (min)')
    axes[0].set_title('A  One descriptive summary per phase',loc='left',fontsize=9);axes[1].set_title('B  Shared cooking and scored coverage',loc='left',fontsize=9)
    for ax in axes:tidy(ax);ax.grid(axis='x',alpha=.15)
    fig.suptitle(f'All pair phases: {first+1}-{last} of {len(reg)}',x=.25,ha='left',y=.96,fontsize=14,fontweight='bold')
    fig.text(.25,.91,'RR interval (IBI) phase consistency; identity order, no condition comparison',fontsize=9)
    fig.text(.25,.08,'Blue: union of scored cores. Grey: shared cooking. No error bars from dependent windows.\n'
             'Descriptive working specification; physical clock mapping unverified; final inclusion not frozen.',fontsize=8,linespacing=1.6)
    return fig

METHODS='''# KITCO3 publication figures v1.1.0

This is a visualization update, not a new synchrony estimator or final analysis.
All 71 pair phases remain. All 10,434 currently scored windows are selectable.
The current working window remains 200 s with 40 s context per side, 1 s start
step, and LF 0.04-0.15 Hz. Final scientific specification is not frozen.

## What changed
Heatmaps are removed from these main figures. They were mathematically valid
context displays, but repeated overlapping windows create diagonal structures
that are poor stand-alone evidence of interpersonal phase locking. Prior output
files remain unchanged. No claim is made that heatmaps are generally invalid.
Every ECG-derived IBI is labelled RR interval (IBI); RR is displayed in ms.
One three-page PDF per pair phase integrates the RR/PLV page with a raw/cleaned
ECG page for each partner. Three booklets collect all pair PDFs. Separate PNGs
are written at 300 dpi. The overview includes all phases and missing estimates.

## Main page
Paired RR intervals are plotted with original gaps. Both a 0-1 PLV view and a
shared enlarged view show exactly the same sealed scores; no curve smoothing.
The enlarged limit is the global maximum across all scored windows rounded up
to 0.05, with a minimum limit of 0.25. No score is clipped. This display choice
does not choose an analysis window or inclusion rule. Mean line is descriptive,
not a null or significance threshold. Coverage is union duration, not a sum of
overlapping windows. A schematic explains core, context and midpoint.
PLV is the magnitude of the mean unit phase-difference vector per 200 s core.
The pair-phase summary is mean local PLV, not pooled whole-interaction PLV.

## Window viewers
Open index.html locally, then choose a pair. Each pair has a self-contained
HTML viewer with a selector and slider for EVERY scored window. Initial window
is first by segment/time, never ranked by PLV. The viewer does not run the
estimator: it displays validated Python results. Full 280 s context-filtered
RR signals and a wrapped phase-difference trace are linked to the selected PLV.
Grey shading identifies the 40 s contexts; only the central 200 s enters PLV.
The circular histogram uses 18 bins of 20 degrees; radial values are fractions
of the 800 core samples. Its radial scale adapts to the selected distribution,
is numerically labelled, and must not be used for cross-window visual effect
size comparisons. Compare the labelled PLV instead. Histograms are not
significance tests and samples are autocorrelated. No pooled phase histogram
is used as a substitute for mean local PLV. Each full-precision PLV is checked
against sealed Script05 to tolerance 1e-8. Float32 display phases reproduce PLV
within 1e-7. Nothing is temporally downsampled in context diagnostics.
Viewers work offline in a current browser supporting DecompressionStream
(current Chrome recommended). No external scripts, network, or server needed.
Use static PDFs for sharing scientific figures; HTML is for detailed inspection.

## Raw versus cleaned ECG
All 142 original ECG CSV hashes must match the frozen source inventory embedded
in the script. Read-only source paths come from original Script02 metadata.
Full-recording cleaning is reconstructed, per recorded processing segment,
using original design rate, powerline policy and NeuroKit2 0.2.13. Script02 did
not save cleaned arrays, so this is a documented reconstruction, not a claim of
byte-identical recovery of original cleaned samples. No peak detection/correction
is rerun. Final Script03 peak source indices, values and timestamps are checked.
Green circles show trusted peaks; purple crosses untrusted retained peaks.
Withheld spans are grey. The full-cooking overview uses per-bin min/max source
samples for display only. Three fixed 10 s sections centred at 25%, 50%, 75%
of cooking show all raw samples. They are not selected for clean morphology or
high PLV. Raw/cleaned plots share amplitude limits within a row, but amplitude
limits may vary between rows. Units are exported units, not assumed microvolts.
These pages are QC displays, not independent human adjudication or new QC rules.

## Limitations
No sensor-clock offset/drift is calibrated, fitted or corrected. RR phase is an
operational band-limited measure, not proof of one physiological oscillator.
No arbitrary low/moderate/high synchrony thresholds, condition p-values, window
SEM/CI, surrogate tests or final cohort freeze are added. Positive PLV alone
does not establish interpersonal coupling. Reference and statistical analyses
remain separate work. Scientific results are unchanged by this plotting update.

## Run on Emma's Mac
Save this script and original 05_KITCO3_PLV_feasibility_v1_0_0_EG.py in:
/Volumes/One Touch/KITCO3_drive/scripts
Run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05b_KITCO3_PLV_publication_figures_v1_1_0_EG.py"

Defaults read outputs/02_ecg_processing_v1_1_0, outputs/03_ecg_QC_v2_1_0,
outputs/04_dyadic_signals_v1_0_0, outputs/05_PLV_feasibility_v1_0_0 and data.
Output is NEW outputs/05b_PLV_publication_figures_v1_1_0. No old outputs deleted.
Options: --project-root, --script02-dir, --script03-dir, --script04-dir,
--script05-dir, --raw-data-root, --processing-script, --output-dir.
Requires numpy, pandas, scipy, matplotlib and neurokit2==0.2.13.
The pinned NeuroKit version is only needed for reconstructed ECG cleaning.
There is no browser or JavaScript dependency to install. All HTML is standalone.
Completion status is written last. Input and output hashes record provenance.

## Output organisation
pairs/PAIR.pdf: RR/PLV summary + partner 1 ECG + partner 2 ECG (three pages).
figures/PAIR_RR_PLV.png and PAIR_partner1/2_ECG.png: 300 dpi pages.
05b_pairs_part_01/02/03.pdf: 72/72/69 pages, preserving identity order.
05b_overview.pdf: three pages covering all 71 phases.
viewers/PAIR.html and index.html: interactive access to every scored window.
05b_figure_index.csv, 05b_RUN_STATUS.json and this README: documentation/audit.

References:
Aydore et al. 2013, doi:10.1016/j.neuroimage.2013.02.008 (PLV properties/bias).
Matsuki et al. 2023, doi:10.1038/s41598-023-30405-5 (Hilbert phase limitations).
These references do not uniquely validate the present band or window length.
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    for name in ['script02-dir','script03-dir','script04-dir','script05-dir','raw-data-root','processing-script','output-dir']:p.add_argument('--'+name,type=Path)
    a=p.parse_args();root=a.project_root.expanduser().resolve()
    src2=(a.script02_dir or root/'outputs/02_ecg_processing_v1_1_0').resolve()
    src3=(a.script03_dir or root/'outputs/03_ecg_QC_v2_1_0').resolve()
    src4=(a.script04_dir or root/'outputs/04_dyadic_signals_v1_0_0').resolve()
    src5=(a.script05_dir or root/'outputs/05_PLV_feasibility_v1_0_0').resolve()
    raw=(a.raw_data_root or root/'data').resolve();out=(a.output_dir or root/'outputs/05b_PLV_publication_figures_v1_1_0').resolve()
    processor=(a.processing_script or Path(__file__).with_name('05_KITCO3_PLV_feasibility_v1_0_0_EG.py')).resolve()
    require(not out.exists(),'Output exists. Preserve it and choose a new --output-dir.')
    require(processor.is_file() and sha(processor)==PROCESSING_SHA,'Need unchanged original Script05 v1.0.0 beside this file')
    import neurokit2 as nk
    require(nk.__version__=='0.2.13','For original cleaning reconstruction, use neurokit2==0.2.13')
    proc=load_module(processor,'original05');up,reg,segments,data=proc.load_inputs(src4)
    state=json.loads((src5/'05_RUN_STATUS.json').read_text())
    require(state['status']=='CONDITIONAL_PLV_FEASIBILITY_COMPLETE' and state['script_sha256']==PROCESSING_SHA,'Unexpected Script05 release')
    require(state['source_status_sha256']==sha(src4/'04_RUN_STATUS.json'),'Script05 used a different Script04 export')
    hashes={str(processor):sha(processor),str(src4/'04_RUN_STATUS.json'):sha(src4/'04_RUN_STATUS.json'),str(src5/'05_RUN_STATUS.json'):sha(src5/'05_RUN_STATUS.json')}
    for src,seals in [(src4,up['output_sha256']),(src5,state['output_sha256'])]:
        for n,h in seals.items():
            f=(src/n).resolve();require(f.is_relative_to(src) and f.is_file() and sha(f)==h,'Changed upstream file: '+n);hashes[str(f)]=h
    handoff=src3/'03_STEP04_HANDOFF.json';require(sha(handoff)==up['source_handoff_sha256'],'Script03 handoff differs from Script04 source')
    hashes[str(handoff)]=sha(handoff);h3=json.loads(handoff.read_text())
    for n,h in h3['output_sha256'].items():
        f=src3/'dataset'/n;require(f.is_file() and sha(f)==h,'Script03 dataset changed: '+n);hashes[str(f)]=h
    peaks=pd.read_csv(src3/'dataset/03_reviewed_peak_times.csv.gz')
    withheld=pd.read_csv(src3/'dataset/03_sample_anchored_withheld_regions.csv')
    windows=pd.read_csv(src5/'05_working_window_diagnostics.csv')
    require(windows.status.eq('CONDITIONAL_DIAGNOSTIC_SCORE').all(),'Unscored or degenerate contexts need separate review')
    require(not windows.duplicated(['pair_id','paired_segment_id','core_start_s']).any(),'Duplicate window')
    require(windows.PLV.between(0,1).all(),'Invalid PLV')
    sums=pd.read_csv(src5/'05_pair_phase_method_summaries.csv');sums=sums.loc[sums.family.eq('working_200s_40s')].set_index('pair_id')
    require(sums.index.is_unique and set(sums.index)==set(reg.pair_id),'Incomplete pair summaries')
    require(set(windows.pair_id).issubset(set(reg.pair_id)),'Orphan scored windows')
    reg=reg.sort_values('pair_id').reset_index(drop=True);zoom=min(1.,max(.25,math.ceil(float(windows.PLV.max())*20)/20))
    require(set(SOURCE_HASHES)==set(reg.recording_uid_1)|set(reg.recording_uid_2),'Frozen raw inventory differs from 71 pair phases')
    # Read source timing from original record metadata, never reconstruct clocks.
    source={}
    for r in reg.itertuples(index=False):
        for uid in [r.recording_uid_1,r.recording_uid_2]:
            f=src2/'records'/uid/'02_result.json';require(f.is_file(),'Missing original Script02 metadata: '+str(f))
            v=json.loads(f.read_text());require(v['recording_uid']==uid,'Metadata identity mismatch')
            require(abs(v['cooking_interval_duration_s']-r.shared_cooking_s)<2e-5,'Cooking duration mismatch')
            source[uid]=type('Source',(),v)
    out.mkdir(parents=True)
    for name in ['pairs','figures','viewers']: (out/name).mkdir()
    atomic_text(out/'05b_README.md',METHODS);styling();rows=[];max_error=0.;nwindows=0
    inputs={'src2':src2,'raw':raw,'hashes':hashes}
    for first in range(0,len(reg),24):
        with book(out/f'05b_pairs_part_{first//24+1:02d}.pdf') as b:
            for row_num,r in enumerate(reg.iloc[first:first+24].itertuples(index=False),first):
                s=sums.loc[r.pair_id];w=windows.loc[windows.pair_id.eq(r.pair_id)]
                require(len(w)==s.windows,'Window count differs from sealed summary')
                if len(w):require(abs(w.PLV.mean()-s.mean_local_PLV)<1e-8,'Mean PLV mismatch')
                coverage=sum(y-x for x,y in spans([(v,v+200) for v in w.core_start_s]))
                require(abs(coverage-s.core_union_s)<1e-7,'Scored coverage mismatch')
                seg=segments.loc[segments.pair_id.eq(r.pair_id)];g=data.get(r.pair_id)
                packed,order,error=phase_payload(proc,r,w,g,seg);max_error=max(max_error,error);nwindows+=len(order)
                viewer(out/'viewers'/f'{r.pair_id}.html',r,packed,order,g,zoom)
                with book(out/'pairs'/f'{r.pair_id}.pdf') as pair_book:
                    fig=main_figure(r,s,w,g,seg,zoom);savefig(fig,out/'figures'/f'{r.pair_id}_RR_PLV.png');pair_book.savefig(fig,dpi=300);b.savefig(fig,dpi=300);plt.close(fig)
                    for j,uid in enumerate([r.recording_uid_1,r.recording_uid_2],1):
                        fig=ecg_figure(uid,r.pair_id,source[uid],peaks,withheld,nk,inputs)
                        savefig(fig,out/'figures'/f'{r.pair_id}_partner{j}_ECG.png');pair_book.savefig(fig,dpi=300);b.savefig(fig,dpi=300);plt.close(fig)
                status='SCORED' if len(w) else ('NO_PAIRED_RR_GRID' if g is None else 'INSUFFICIENT_CONTINUOUS_CONTEXT')
                rows.append({'pair_id':r.pair_id,'status':status,'windows':len(w),'mean_local_PLV':s.mean_local_PLV,
                             'shared_cooking_s':r.shared_cooking_s,'scored_core_union_s':s.core_union_s,
                             'pdf':f'pairs/{r.pair_id}.pdf','viewer':f'viewers/{r.pair_id}.html',
                             'book':f'05b_pairs_part_{first//24+1:02d}.pdf','first_book_page':(row_num-first)*3+1})
                print(f'Complete {len(rows)}/71: {r.pair_id}; {len(w)} windows; both ECGs checked',flush=True)
    require(nwindows==len(windows),'Not every scored window was reproduced')
    with book(out/'05b_overview.pdf') as b:
        for first in range(0,len(reg),24):
            fig=overview(reg,sums,first,min(first+24,len(reg)));b.savefig(fig);savefig(fig,out/'figures'/f'05b_overview_{first//24+1:02d}.png');plt.close(fig)
    index=pd.DataFrame(rows);index.to_csv(out/'05b_figure_index.csv',index=False)
    links=''.join(f'<tr><td>{html.escape(v["pair_id"])}</td><td>{v["windows"]}</td><td>{v["status"]}</td><td><a href="{v["pdf"]}">Three-page PDF</a></td><td><a href="{v["viewer"]}">Every scored window</a></td></tr>' for v in rows)
    atomic_text(out/'index.html','<!doctype html><meta charset="utf-8"><title>KITCO3 RR and PLV figures</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:15px}td,th{padding:8px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style><h1>All 71 pair phases: RR interval (IBI) and PLV</h1><p>Descriptive working estimates. Clock mapping unverified; final inclusion not frozen. <a href="05b_overview.pdf">All-phase overview</a></p><table><thead><tr><th>Pair phase</th><th>Overlapping windows</th><th>Availability</th><th>Results and ECG</th><th>Window diagnostics</th></tr></thead><tbody>'+links+'</tbody></table>')
    for n,h in hashes.items():require(sha(Path(n))==h,'Input changed during plotting: '+n)
    result={'script':Path(__file__).name,'version':VERSION,'script_sha256':sha(Path(__file__)),
            'status':'ALL_PHASE_PUBLICATION_FORMAT_PLOTS_COMPLETE','paired_phases':len(reg),'scored_phases':int((index.windows>0).sum()),
            'scored_windows_reproduced':nwindows,'ECG_recordings_source_verified':len(source),'maximum_PLV_reproduction_error':max_error,
            'shared_enlarged_PLV_ylim':[0,zoom],'heatmaps_included':False,'ECG_cleaning':'RECONSTRUCTED_FROM_ORIGINAL_SCRIPT02_SETTINGS',
            'ECG_peak_detection_rerun':False,'new_QC_exclusions':False,'condition_effect_computed':False,'physical_clock_verified':False,'final_synchrony_dataset_frozen':False,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'neurokit2':nk.__version__},
            'input_sha256':hashes,'output_sha256':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and '.writing' not in f.name}}
    atomic_text(out/'05b_RUN_STATUS.json',json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','output_sha256']},indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('PLOTTING STOPPED: '+str(e)+'. Failed runs do not publish a completion status.',file=sys.stderr);sys.exit(1)
