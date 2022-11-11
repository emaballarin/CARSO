#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TA2ATAdapter:
    def __init__(self, attack):
        self.attack = attack

    def perturb(self, x, y):
        return self.attack(x, y)
