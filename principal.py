# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 

import sr6 as s

s.bm = s.Bitmap(900,900)
#                                 translate    scale        eye       up      center      color     luz
s.bm.load('esferatriangulada.obj', (0,0,0), (0.8,0.8,0.8), (1,1,1), (0,1,0), (0,0,0), (0,255,0), (1,1,1))
s.glFinish("sphere")