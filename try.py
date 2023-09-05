import torch
from utils.general import box_iou,xywh2xyxy,bbox_iou
import torchvision
import argparse
import yaml
device='cuda:0'
from utils.datasets import create_dataloader




import random
def _make_grid(nx=10, ny=10):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
grid = _make_grid(3,3)
x=torch.ones([2,3,3,3,5])

y = x.sigmoid()
p=[torch.rand([2,3,10,10,8]),torch.rand([2,3,20,20,8])]#,torch.rand([2,3,15,15,8])]#,torch.rand([2,3,10,10,8])]
cls=torch.Tensor([[0,1],[0,1],[0,3],[0,1],[1,1],[1,2],[1,2]])
target=torch.cat((cls,torch.rand([7,4])),dim=1)
def build_targets( p, targets):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 2, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    print(targets.shape)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    print(targets.shape)
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    anchors_=torch.tensor([[[3,4],[5,6]],[[5,6],[3,4]]])
    for i in range(2):#layer
        anchors = anchors_[i]
        print(anchors)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain #
        #print(gain)

        # Match targets to anchors
        t = targets * gain #in 20*20 imgs
        #print(t)
        if nt:
            # Matches
            #print(anchors[:,None])
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio#
            print(r.shape,1./r,torch.max(r, 1. / r),torch.max(r, 1. / r).max(2))
            #return
            j = torch.max(r, 1. / r).max(2)[0] < 4 # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            print(j,k,l,m)
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            print(t)
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            print(j,t,offsets)
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices
        print(gij)

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

    
#build_targets(p,targets=target)
pbox=torch.rand([10,4])*10
tbox=torch.rand([10,4])*10
pbox=torch.tensor([[3,8,5,6],[1,3,4,7]])
x=torch.tensor([[2,2,5,6],[5,3,4,2],[5,5,4,7]])
iouv = torch.linspace(0.35, 0.95, 13)
ious=torch.Tensor([0.9,0.3,0.7,0.1])
ti=torch.Tensor([0,1,4])
pi=[1,4,5,7]
i=[0,2,0,1]
detected_set = set() 
detected = []
correct = torch.zeros(10, 13, dtype=torch.bool)
for j in (ious > iouv[0]).nonzero(as_tuple=False):
    print(j)
    d = ti[i[j]]  # detected target
    if d.item() not in detected_set:
        detected_set.add(d.item())
        detected.append(d)
        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
        if len(detected) == 3:  # all targets already located in image
            break
stats=[]
stats.append((correct, x[1], x[0],pbox[:,0] ))
stats.append((correct, x[2], x[0],pbox[:,1] ))
import numpy as np
stats = [np.concatenate(x, 0) for x in zip(*stats)]
#print(stats[0])
#print(ious[np.argsort(-ious)])
#print(int(#np.random.random()*10))
print(torch.rand(pbox.shape))