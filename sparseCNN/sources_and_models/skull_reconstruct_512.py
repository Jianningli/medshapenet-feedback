#######################################__________________________________________________--
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import gc, metrics, nrrd, torch, data_skull, argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import MinkowskiEngine as ME
from time import time
import torch
from weight_initializer import Initializer
import torch.nn.init as init
#  torch.cuda.is_available = lambda : False  # To force CPU usage across ME
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--torch_seed", type=int, default=105)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--weights", type=str, default="skull_completion_512.pth")


class CompletionNet(nn.Module):
    #ENC_CHANNELS = [22, 32, 32, 128, 156, 256, 388]
    ENC_CHANNELS = [2, 2, 4, 4, 4, 4, 16]
    DEC_CHANNELS = ENC_CHANNELS

    def __init__(self):
        nn.Module.__init__(self)
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
        )
        #
        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(dec_ch[3], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3)

        self.final_out = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiSigmoid(),
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            target = torch.zeros(len(out), dtype=torch.bool, device=device)
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def get_keep_vec(self, cls, res):
        """
        To ensure that a sparse tensor can safely be converted to the dense one.
        """
        a = (cls.F > 0).squeeze()
        b = (cls.C[:, 1] < res[2]).squeeze()
        c = (cls.C[:, 2] < res[3]).squeeze()
        d = (cls.C[:, 3] < res[4]).squeeze()
        ab = torch.logical_and(a, b)
        abc = torch.logical_and(ab, c)
        abcd = torch.logical_and(abc, d)
        return abcd

    def forward(self, partial_in, target_key):
        crit = nn.BCEWithLogitsLoss()
        loss = 0
        out_cls, targets = [], []
        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        # ##################################################
        # # Decoder 64 -> 32
        # ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64) + enc_s32
        # Add encoder features
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        target = 0
        if target_key != 0:
            target = self.get_target(dec_s32, target_key)
            loss += crit(dec_s32_cls.F.squeeze(), target.type(dec_s32_cls.F.dtype).to(device))
        if self.training:
            keep_s32 += target

        # Remove voxels s32
        dec_s32 = self.pruning(dec_s32, keep_s32)
        del keep_s32, target, dec_s32_cls
        gc.collect()
        torch.cuda.empty_cache()

        # ##################################################
        # # Decoder 32 -> 16
        # ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32) + enc_s16

        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        target = 0
        if target_key != 0:
            target = self.get_target(dec_s16, target_key)
            loss += crit(dec_s16_cls.F.squeeze(), target.type(dec_s16_cls.F.dtype).to(device))

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)
        del dec_s16_cls, target, keep_s16
        gc.collect()
        torch.cuda.empty_cache()

        ##################################################
        # Decoder 16 -> 8
        ##################################################

        dec_s8 = self.dec_block_s16s8(dec_s16) + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        target = 0
        if target_key != 0:
            target = self.get_target(dec_s8, target_key)
            loss += crit(dec_s8_cls.F.squeeze(), target.type(dec_s8_cls.F.dtype).to(device))

        if self.training:
            keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)
        del dec_s8_cls, keep_s8, target
        gc.collect()
        torch.cuda.empty_cache()

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8) + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        target = 0
        if target_key != 0:
            target = self.get_target(dec_s4, target_key)
            loss += crit(dec_s4_cls.F.squeeze(), target.type(dec_s4_cls.F.dtype).to(device))

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        del dec_s4_cls, keep_s4, target
        gc.collect()
        torch.cuda.empty_cache()

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4) + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        target = 0
        if target_key != 0:
            target = self.get_target(dec_s2, target_key)
            loss += crit(dec_s2_cls.F.squeeze(), target.type(dec_s2_cls.F.dtype).to(device))

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)
        del dec_s2_cls, keep_s2, target
        gc.collect()
        torch.cuda.empty_cache()

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2) + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        if target_key != 0:
            loss += crit(dec_s1_cls.F.squeeze(), self.get_target(dec_s1, target_key).type(dec_s1_cls.F.dtype).to(device))

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target
        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, (dec_s1_cls.F > 0).squeeze())
        del dec_s1_cls

        torch.cuda.empty_cache()
        dec_s1 = self.final_out(dec_s1)
        return out_cls, targets, dec_s1, loss


def get_dense(net, to_prune, res):
    out = ME.MinkowskiPruning()(to_prune, net.get_keep_vec(to_prune, res))
    return ME.MinkowskiToDenseTensor(res)(out)


def get_numpys(sin, sout, ground_truth, res):

    dense_in = ME.MinkowskiToDenseTensor(res)(sin).detach().cpu().squeeze().numpy()
    dense_out = get_dense(net, sout, res).detach().cpu().squeeze().numpy()
    truth = ME.SparseTensor(
        features=torch.ones((len(ground_truth[0]), 1)),
        coordinates=ME.utils.batched_coordinates(ground_truth),
        device="cpu")
    dense_truth = ME.MinkowskiToDenseTensor(res)(truth).squeeze().numpy()
    del truth
    return dense_in, dense_out, dense_truth


TRAIN_LOSS = []
EVAL_LOSS = []
DICE = []
BORDER_DICE = []
HAUSD = []
HAUSD_95 = []


def train(net, train_dataloader, valid_dataloader, device, config):
    optimizer = optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)
    net.train()
    #torch.cuda.empty_cache()
    for epoch in range(config.epochs):
        if epoch > 0:
            #net.eval()
            #bce, ds, bds, hd_95, hd = eval(net, valid_dataloader, device, epoch)
            #EVAL_LOSS.append(bce)
            #DICE.append(ds)
            #BORDER_DICE.append(bds)
            #HAUSD.append(hd)
            #HAUSD_95.append(hd_95)
            #data_skull.plot_loss(TRAIN_LOSS, EVAL_LOSS)
            #data_skull.plot_dice_metrics(DICE, BORDER_DICE)
            #data_skull.plot_hd_metrics(HAUSD, HAUSD_95)
            # with open(f"metrics_{epoch}.txt", "w+") as output:
            #     output.write(str([ds, bds, hd_95, hd]))
            net.train()
        # elif epoch != 0 and len(EVAL_LOSS) > 0:
        #     EVAL_LOSS.append(EVAL_LOSS[len(EVAL_LOSS)-1])

        if epoch % 2 == 0 and epoch != 0:
            print(f"SAVING at {epoch}th epoch")
            torch.save(net.state_dict(), config.weights)
        EPOCH_LOSS = []

        #len: 100
        print('len:',len(train_dataloader))
        #<torch.utils.data.dataloader.DataLoader object at 0x7efde8a3c610>
        print(train_dataloader)

        
        for i in range(len(train_dataloader)):
            s = time()
            data_dict = next(iter(train_dataloader))
            #print('******')
            #print('defective',len(data_dict['defective']))
            
            '''
             tensor([[0, 10, 30, 15],
                    [ 0, 10, 30, 16],
                    [ 0, 10, 30, 17],
                    ...,
                    [ 0, 51, 41, 18],
                    [ 0, 51, 41, 19],
                    [ 0, 51, 41, 20]], dtype=torch.int32)
            '''
            #print('******')
            CUR_RES = torch.Size(data_dict["shape"][0])
            d = time() - s

            optimizer.zero_grad()
            #[[1, 1, 64, 64, 32]]
            #print(data_dict["shape"][0])

            in_feat = torch.ones((len(data_dict["defective"]), 1))
            sin = ME.SparseTensor(
                features=in_feat,
                coordinates=data_dict["defective"],
                device=device,
            )

            # Generate target sparse tensor
            cm = sin.coordinate_manager
            target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(data_dict["complete"]).to(device),
                string_id="target",
            )
            #print('complete:',data_dict["complete"])
            
            '''
            complete: [tensor([[10., 30., 15.],
                               [10., 30., 16.],
                               [10., 30., 17.],
                                ...,
                               [51., 41., 18.],
                               [51., 41., 19.],
                               [51., 41., 20.]])]
            '''

            # Generate from a dense tensor
            print(f"_________________START[{epoch}|{i}]___________________________")
            #print('summary:',torch.cuda.memory_summary('cuda:0'))
            out_cls, targets, sout, losst = net(sin, target_key)
            #print('summary',torch.cuda.memory_summary('cuda:0'))
            print(len(sin), "->", len(sout), "/", len(data_dict["complete"][0]))

            losst.backward()
            optimizer.step()

            #if i == len(train_dataloader)-1:
            #    dense_in, dense_out, dense_truth = get_numpys(sin, sout, data_dict["complete"], CUR_RES)
            #    print("Plotting")
            #    data_skull.plot_slice_in_out_truth(dense_in, len(sin), dense_out, len(sout), dense_truth,
            #                                         len(data_dict["complete"][0]), 98, 1000 * epoch + i, 0.5)
            #    del dense_in, dense_out, dense_truth

            t = time() - s
            print(f"Iter: {i}, Loss: {losst.item():.3f}, Data Loading Time: {d:.3f}, Total Time: {t:.3f}")
            print(f"_________________FINISH[{epoch}|{i}]__________________________")
            EPOCH_LOSS.append(losst.item())

            del losst, sout, sin, targets, out_cls, target_key
            gc.collect()
            torch.cuda.empty_cache()
            net.train()
        TRAIN_LOSS.append(np.mean(EPOCH_LOSS))

    #data_skull.plot_loss(TRAIN_LOSS, EVAL_LOSS)


def eval(net, dataloader, device, it):
    print("____________________________________________EVAL____________________________________________")
    net.eval()
    EVAL_LOSS = []
    DS = []
    HD_95 = []
    HD = []
    BDS = []
    with torch.no_grad():
        for j in range(len(valid_loader)):
            s = time()
            data_dict = next(iter(dataloader))
            d = time() - s

            in_feat = torch.ones((len(data_dict["defective"]), 1))
            sin = ME.SparseTensor(
                features=in_feat,
                coordinates=data_dict["defective"],
                device=device,
            )
            # Generate target sparse tensor
            cm = sin.coordinate_manager

            target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(data_dict["complete"]).to(device),
                string_id="target",
            )
            CUR_RES = torch.Size(data_dict["shape"][0])
            # Generate from a dense tensor
            print(f"_________________START[{it}]___________________________")
            out_cls, targets, sout, losst = net(sin, target_key)
            print(len(sin), "->", len(sout), "/", len(data_dict["complete"][0]))
            print("Plotting")
            dense_in = ME.MinkowskiToDenseTensor(CUR_RES)(sin).detach().cpu().squeeze().numpy()
            dense_out = get_dense(net, sout, CUR_RES).cpu().squeeze().numpy()

            truth = ME.SparseTensor(
                features=torch.ones((len(data_dict["complete"][0]), 1)),
                coordinates=ME.utils.batched_coordinates(data_dict["complete"]),
                device="cpu")

            dense_truth = ME.MinkowskiToDenseTensor(CUR_RES)(truth).squeeze().numpy()
            del truth
            predicted_implant = data_skull.filter_implant(dense_out, dense_in)

            DS.append(metrics.dc(predicted_implant, dense_truth - dense_in))
            BDS.append(metrics.bdc(predicted_implant, dense_truth - dense_in, dense_in))
            HD_95.append(metrics.hd95(predicted_implant, dense_truth-dense_in))
            HD.append(metrics.hd(predicted_implant, dense_truth - dense_in))

            data_skull.plot_slice_in_out_truth(dense_in, len(sin), dense_out, len(sout), dense_truth,
                                                 len(data_dict["complete"][0]), 98, it, True)
            t = time() - s
            del dense_truth, dense_in, dense_out, predicted_implant
            print(f"Iter: {it}, Loss: {losst.item():.3f}, Data Loading Time: {d:.3f}, Tot Time: {t:.3f}")
            print(f"_________________FINISH[{it}]__________________________")
            EVAL_LOSS.append(losst.item())
            del losst, sout, sin, targets, out_cls
            gc.collect()
            torch.cuda.empty_cache()
    bce = np.mean(EVAL_LOSS)
    ds = np.mean(DS)
    bds = np.mean(BDS)
    hd_95 = np.mean(HD_95)
    hd = np.mean(HD)
    print("AVG BCE:",  bce)
    print(EVAL_LOSS)
    print("AVG Dice Score:", ds)
    print(DS)
    print("AVG BDS:",  bds)
    print(BDS)
    print("AVG HD_95:",  hd_95)
    print(HD_95)
    print("AVG HD:",  hd)
    print(HD)
    return bce, ds, bds, hd_95, hd


def test(net, dataloader, device, it):
    print("____________________________________________TEST____________________________________________")
    #net.eval()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            i=0
            data_dict = d
            #print('data_dict[0].squeeze():',data_dict[0].squeeze())
            in_feat = torch.ones((len(data_dict[0].squeeze()), 1))
            sin = ME.SparseTensor(
                features=in_feat,
                coordinates=data_dict[0].squeeze(),
                device=device,
            )

            CUR_RES = torch.Size(data_dict[1])
            ORIG_RES = data_dict[2]

            print(CUR_RES)
            print(f"_________________START[{it}]___________________________")
            out_cls, targets, sout, losst = net(sin, 0)
            #print('out_cls',out_cls)
            print(len(sin), "->", len(sout))
            #coord, feats = ME.utils.sparse_quantize(sout.C.cpu(), sout.F.cpu())
            #print('coord',coord)
            #print('feats',feats)

            sout.F[:]=1


            string = str(j).zfill(3)
            if j!=61:
                data, header = nrrd.read(data_skull.TEST_EDGES_ROOT +'/defective_skull/'+f"/{string}.nrrd")
            #dense_in = ME.MinkowskiToDenseTensor(CUR_RES)(sin).detach().cpu().squeeze().numpy()
            #dense_out = get_dense(net, sout, CUR_RES).cpu().squeeze().numpy()

            dense_out = ME.MinkowskiToDenseTensor(CUR_RES)(sout).detach().cpu().squeeze().numpy()    

            implant=dense_out
            #implant = data_skull.filter_implant(dense_out, dense_in)

            # Padding can be avoided if you chose to simply zero parts of the volume
            #npad = [(0, 0)] * implant.ndim
            #s = data_dict[3].item()
            #npad[2] = ((ORIG_RES[4] - CUR_RES[4]).item() - s, s)
            #implant = np.pad(implant, npad, mode='constant', constant_values=0)
            #out = implant + data

            string = str(j).zfill(3)

            nrrd.write(data_skull.TEST_EDGES_ROOT + f"/Predictions/{string}IMPLANT.nrrd", implant.astype('int32'), header)
            #nrrd.write(data_skull.TEST_EDGES_ROOT + f"/Predictions/{string}IN.nrrd", data, header)
            #nrrd.write(data_skull.TEST_EDGES_ROOT + f"/Predictions/{string}OUT.nrrd", out.astype('int32'), header)
            #del implant, dense_in, dense_out
            print(f"_________________FINISH[{it}]__________________________")
            del losst, sout, sin, targets, out_cls
            gc.collect()
            torch.cuda.empty_cache()





if __name__ == '__main__':
    config = parser.parse_args()
    np.random.seed(210)
    torch.manual_seed(config.torch_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for  test on cpu
    #device=torch.device("cpu")

    train_loader, valid_loader = data_skull.get_train_valid_loader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
    )

    net = CompletionNet()
    Initializer.initialize(model=net, initialization=init.normal)
    print("#Parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("Using: ", device)
    net.to(device)
    if config.pretrained:
        print('succefully loaded the checkpoint...')
        weights = torch.load(config.weights)
        net.load_state_dict(weights)
    train(net, train_loader, valid_loader, device, config)
    #eval(net, valid_loader, device, 9999)
    
    #test_loader, additional = data_skull.get_testing_dataloader()
    
    #test_loader = data_skull.get_testing_dataloader()
    #net.eval()
    #test(net, test_loader, device, 101010)
    #test(net, additional, device, 101010)


