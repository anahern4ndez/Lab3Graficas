# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 

import shaders as s

s.bm = s.Bitmap(500,500)
s.Estrellas()
#                                 translate    scale      rotate      eye       up      center      color     luz
#s.bm.load('esferatriangulada.obj', (0,0,0), (0.8,0.8,0.8), (0,0,0), (0,0,5), (0,1,0), (0,0,0), (1.2,0.1,0.7))
s.bm.load('esfera.obj', (0,0,0), (0.8,0.8,0.8), (0,0,0), (0,0,5), (0,1,0), (0,0,0), (0.5,0.1,1))
s.glFinish("sphere")