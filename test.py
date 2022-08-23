

# -*- coding: utf-8 -*-
"""
"""

import argparse
import sys
import os
import subprocess


import slack


import math

parser = argparse.ArgumentParser(description='PyTorch local error training')
parser.add_argument('--test', action="store_true",
                    help='store test directory')
parser.add_argument('--noslack', action="store_true",
                    help='store test directory')


parser.add_argument('--gpu', default='0', type=str,
                        help='num_exit for each earlt exit')
parser.add_argument('--tsubame', action="store_true",
                    help='if tsubame')
parser.add_argument('--tsubame-id', default=-1, type=int,
                    help='tsubame id for ')
args = parser.parse_args()


def printy(s):
    print("\033[36m"+s+"\033[0m")

def do_python_eval(command):
    printy(command)
    subprocess.call(command.split())

def do_python(command, base, savedir, multiple=False):
    if args.test:
        savedir = "test"

    # command += " --save %s  2>&1 | tee %s/result.log " % (savedir,savedir)
    command += " --name %s " % (savedir)
    printy(command)

    if not (os.path.exists(base+savedir)) or args.test or multiple:
        if args.test:
            try:
                subprocess.call(command.split())
            except:
                pass
        else:
            subprocess.call(command.split())

    else:
        print("Skipped.")


def main():

    if args.tsubame :
        DATASET = '/gs/hs0/tga-artic'
        DISTILDATASET = '/gs/hs0/tga-artic'
        WORKER = '26'
        CACHE = " --data-cache %s "%os.environ['TMPDIR']
    else:
        DATASET = '/ldisk'
        DISTILDATASET = '/work'
        WORKER = '8'
        CACHE = ''



    BASE = './runs/resnet50-sc-unsigned/'
    COMMAND = "python3 main.py --config configs/largescale/subnetonly/resnet50-sc-unsigned.yaml --name default --data %s/Shared/Datasets/ILSVRC/ --workers %s %s --multigpu %s "%(DATASET, WORKER, CACHE, args.gpu)



    SUB = ' --set ImageNet '

    SUB += ' --arch ResNet50_p2  --init randtest_xor16_ko '
    SUB += '  --epochs 150   '
    SUB += ' --print-freq 1000 '

    topklist = ['0.3']

    for i in range(len(topklist)):

        # topk = topklist[i]
        # SUB2 = " --prune-rate %s "%topk
        # SAVEDIR='2830-no-distil/topk-%s'%(topk)
        # do_python(COMMAND+SUB+SUB2,BASE,SAVEDIR, multiple=False)

        topk = topklist[i]
        SUB2 = " --prune-rate %s "%topk
        SUB2 += ' --bn-type AffineBatchNorm '
        SAVEDIR='2830-no-distil/topk-%s-bn'%(topk)
        do_python(COMMAND+SUB+SUB2,BASE,SAVEDIR, multiple=False)


    SUB = ' --set ImageNet '
    SUB += ' --arch ResNet50_p2  --init randtest_xor16_ko '
    SUB += ' --epochs 200   '
    SUB += ' --print-freq 1000 '

    # 続きに対して量子化
    for i in range(len(topklist)):

        # topk = topklist[i]
        # SUB2 = " --prune-rate %s "%topk
        # SUB2 += ' --resume  ./runs/resnet50-sc-unsigned/2830-no-distil/topk-%s/prune_rate=%s/checkpoints/model_best.pth'%(topk,topk)
        # SUB2 += ' --act-bw block8 '
        # SAVEDIR='2830-no-distil/topk-%s_BFP8'%(topk)
        # do_python(COMMAND+SUB+SUB2,BASE,SAVEDIR, multiple=False)


        topk = topklist[i]
        SUB2 = " --prune-rate %s "%topk
        SUB2 += ' --resume  ./runs/resnet50-sc-unsigned/2830-no-distil/topk-%s-bn/prune_rate=%s/checkpoints/model_best.pth'%(topk,topk)
        SUB2 += ' --bn-type AffineBatchNorm '
        SUB2 += ' --act-bw block8 '
        SAVEDIR='2830-no-distil/topk-%s-bn_BFP8'%(topk)
        do_python(COMMAND+SUB+SUB2,BASE,SAVEDIR, multiple=False)





    # 量子化 の続きの続き
    for i in range(len(topklist)):
        if (not args.tsubame) or (args.tsubame_id == i):
            topk = topklist[i]

            SUB2 += ' --resume  ./runs/resnet50-sc-unsigned/2803-distil-topk/topk-%s_BFP8_continue/prune_rate=%s/checkpoints/model_best.pth'%(topk,topk)

            SUB2 += ' --act-bw block8 '
            SAVEDIR='2803-distil-topk/topk-%s_BFP8_continue2'%(topk)
            do_python(COMMAND+SUB+SUB2,BASE,SAVEDIR, multiple=False)




if __name__ == "__main__":
    prog = sys.argv

    if args.noslack:
        main()
        exit()

    if args.test == False:
        s = slack.slack(filename=True)
        s.notify('Start: %s'%prog[0])

    try:
        main()
        if args.test == False:
            s.notify('Finish: %s'%prog[0])

    except:
        if args.test == False:
            s.notify('Error: %s'%prog[0])
        else:
            print('Error: %s'%prog[0])





