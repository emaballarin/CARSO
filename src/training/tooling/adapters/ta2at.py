#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TA2ATAdapter:
    def __init__(self, attack, xsize, nchan=1):
        self.attack = attack
        self.nchan, self.xsize = nchan, xsize

    def perturb(self, x, y):
        return self.attack(x.reshape(x.shape[0], self.nchan, *self.xsize), y)
