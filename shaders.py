
# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 
#Modulo donde se guardas las funciones necesarias para el sr5. 

import struct
from collections import namedtuple
from math import *
import perlin as p
from random import randint as ri

#variables globales
Vector2 = namedtuple('Vertex2',['x', 'y'])
Vector3 = namedtuple('Vertex3',['x', 'y', 'z'])
bm = None
vpx = 0 #esquina inferior izquierda del VP (y)
vpy = 0 #esquina inferior izquierda del VP (y)
vpWidth = 0 #ancho del viewport
vpHeight = 0 #altura del viewport
centrox =0 # centro del viewport: coordenada x 
centroy =0 # centro del viewport: coordenada y
x0 = 0
y0 = 0

def sumaVectorial(v0, v1):
    return Vector3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def restaVectorial(v0, v1):
    return Vector3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mulEscalar(v0, k):
    return Vector3(v0.x*k, v0.y*k, v0.z*k)

def prodPunto(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def prodCruz(v0, v1):
    return Vector3(
        v0.y* v1.z - v0.z* v1.y,
        v0.z* v1.x - v0.x* v1.z,
        v0.x* v1.y - v0.y* v1.x
    )
#magnitud del vector
def magnitud(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2) ** 0.5

#vector normalizado
def normalizar(vector):
    longitud = magnitud(vector)
    if not longitud: #if not l
        return Vector3(0,0,0)
    return Vector3(vector.x/longitud, vector.y/longitud, vector.z/longitud)

def ordenarXY(A,B,C):
    xsorted = sorted([A.x, B.x, C.x])
    ysorted = sorted([A.y, B.y, C.y])

    return Vector2(xsorted[0], ysorted[0]), Vector2(xsorted[2], ysorted[2]) #x, y minimo; #x,y maximo

#coordenadas baricentricas 
def barycentric(A, B, C, D):
    cx, cy, cz = prodCruz(
        Vector3(B.x - A.x, C.x- A.x, A.x - D.x),
        Vector3(B.y - A.y, C.y- A.y, A.y - D.y),
    )

   # [cx cy cz] = [u v 1], para que esta igualdad se cumpla:
        # [cx/cz cy/cz cz/cz] = [u v 1]
    u,v,w = 0,0,0
    if cz != 0: # en realizdad cz no puede ser < 1
        u = cx/cz
        v = cy/cz
        w = 1 - (u + v)
    else: 
        u,v,w = -1,-1,-1
    return w,v,u

def Char(c):
    return struct.pack("=c", c.encode('ascii'))

def word(c):
    return struct.pack("=h", c)

def dword(c):
    return struct.pack("=l", c)

def color (r,g,b):
    return bytes([b,g,r])

def glInit():
    pass

def glCreateWindow(width, height):
    r =  Bitmap(width, height)
    return r

# x y y representan el punto en el que esta la esquina inferior izquierda del viewport
# width y height son las dimensiones
# se ingresan coordenadas de -1 a 1
def glViewPort(x,y,width, height):
    global vpx, vpy, vpWidth, vpHeight
    vpx = 0
    vpy = 0
    vpWidth = 0
    vpHeight = 0
    vpx = x
    vpy = y
    vpWidth = width
    vpHeight = height

def glClear():
    global bm
    bm.clear()

#c1 =r      c2=g        c3 =b
def glClearColor(r, g, b):
    global bm
    bm.framebuffer = [
            [
                color(int(r*255), int(g*255), int(b*255)) 
                    for x in range(bm.width)
            ]
            for y in range(bm.height)
        ]
def glVertex(x,y, color):
    global vpx, vpy, centrox, centroy, bm, vpHeight, vpWidth, x0, y0
    centrox = vpx + vpWidth/2 
    centroy = vpy + vpHeight/2
    x *= vpWidth/2
    y *= vpHeight/2
    bm.point(int(centrox+ x), int(centroy+y), color)
    x0 = x
    y0 = y

def glColor(r,g,b):
    cr = int(r*255)
    cg = int(g*255)
    cb = int(b*255)
    bm.point(int(centrox+x0), int(centroy+y0), color(cr,cg,cb))

def glFinish(nombre):
    global vpx, vpy, centrox, centroy, bm, vpHeight, vpWidth, x0, y0
    bm.write(nombre + ".bmp")
    bm = None
    vpx = 0
    vpy = 0
    vpWidth = 0
    vpHeight = 0
    centrox =0
    centroy =0
    x0 = 0
    y0 = 0

def mulMat(A, B):
    #en caso que la matriz A sea un vector, no una matriz (en el caso del vector "aumentado")
    filasA = len(A)
    colA = len(A[0])
    colB = len(B[0])
    try: 
        colA = len(A[0])
    except(TypeError):
        colA = 1
    #en caso que la matriz B sea un vector, no una matriz (en el caso del vector "aumentado")
    try: 
        colB = len(B[0])
    except(TypeError):
        colB = 1

    #creacion de matriz resultado:
    C=[]
    try: 
        for i in range(filasA):
            C.append([0]*colB)

        #multiplicacion de A*B y store en C. 
        for i in range(filasA):
            for j in range(colB):
                for k in range(len(B)):
                    C[i][j] += A[i][k] * B[k][j]
    except(RuntimeError, TypeError, NameError) as error:
        print(error)
    return C

# ==========================================================================
#               CLASE OBJ
# ==========================================================================
#clase que servira para crear el obj

class Obj(object):
    def __init__(self, filename):
        self.vertices =[] 
        self.texvert = []
        self.faces = []
        self.normals = []
        with open(filename) as f:
            self.lines = f.read().splitlines()
        self.read()

    # se realiza la lectura del archivo obj
    def read(self):
        for line in self.lines:
            if line:
                prefix, value = line.split(' ', 1)
                
                #vertices del modelo
                if prefix == "v":
                    self.vertices.append(list(map(float, value.split(' '))))
                
                #vertices de las caras
                elif prefix == "f":
                    self.faces.append([list(map(int, face.split('/'))) for face in value.split(' ')])
                
                elif prefix == "vn":
                    self.normals.append(list(map(float, value.split(' '))))

# ==========================================================================
#               CLASE BITMAP 
# ==========================================================================

class Bitmap(object):
    def __init__(self, width, height):
        self.active_shader = None
        self.width = width
        self.height = height
        self.framebuffer = []
        self.clear()
        glViewPort(1, 1, width-1, height -1)

    def clear(self):
        self.framebuffer = [
            [
                color(0, 0, 0) 
                    for x in range(self.width)
            ]
            for y in range(self.height)
        ]
        self.zbuffer = [
            [-float('inf')
                    for x in range(self.width)
            ]
            for y in range(self.height)
        ]

    def write(self, filename):
        f = open(filename, 'wb')
        
        #file header  (14)
        f.write(Char('B'))
        f.write(Char('M'))
        f.write(dword(54 +self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(54))

        #image header 40
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height *3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        for x in range(self.height):
            for y in range(self.width):
                f.write(self.framebuffer[x][y])
        f.close()

    def point(self,x,y,color = color(255,255,255)):
        self.framebuffer[y][x] = color

#el rotate tiene los angulos medidos en radianes
#    def load(self, filename, matfile, translate =(-0.75,-0.75,-0.5), scale= (1000, 1000, 1000), rotate = (0,0,0)):
    def load(self, filename, translate =(0,0,0), scale= (0.97, 0.97, 0.97), rotate = (0,0,0),
            eye = (0,0.5,0.5), up = (0,1,0), center=(0,0,0), luz=(0,0,1), active_shader = None):
        model = Obj(filename)
        if filename == "esfera.obj":
            self.active_shader = gouradPlanet
        self.loadViewportMatrix()
        self.loadModelMatrix(translate, scale, rotate)
        self.lookAt(Vector3(*eye), Vector3(*up), Vector3(*center))

        #aplicación de luz y material a cada cara encontrada en el modelo
        for face in model.faces:
            vcount = len(face)
            if vcount == 3:
                f1 = face[0][0] -1
                f2 = face[1][0] -1
                f3 = face[2][0] -1

                a = self.transform(model.vertices[f1])
                b = self.transform(model.vertices[f2])
                c = self.transform(model.vertices[f3])                
                
                n1 = face[0][2] -1
                n2 = face[1][2] -1
                n3 = face[2][2] -1

                nA = Vector3(*model.normals[n1])
                nB = Vector3(*model.normals[n2])
                nC = Vector3(*model.normals[n3])
                self.triangle(a,b,c, nA, nB, nC, luz)
    
    def triangle(self, A, B, C, nA, nC, nB, luz):
        
        xy_min, xy_max = ordenarXY(A,B,C)
        #print(xy_min, xy_max)
        for x in range(xy_min.x, xy_max.x + 1):
            for y in range (xy_min.y, xy_max.y + 1):
                #print(x,y)
                w, v, u = barycentric(A,B,C, Vector2(x,y))
                #print(w,v,u)
                if w< 0 or v <0 or u<0:
                    continue
                color = self.active_shader(self, x, y, bar=(w,v,u), normales=(nA, nB, nC), light = Vector3(*luz))
                z = A.z*w + B.z*v  + C.z*u
                if z > self.zbuffer[x][y]:
                    self.point(x,y,color)
                    self.zbuffer[x][y] = z

# ==========================================================================
#                       Métodos de matrices
# ==========================================================================
    
    def transform(self, vertex):
        aumentado = [
            [vertex[0]],
            [vertex[1]],
            [vertex[2]],
            [1.0]
        ]
        #la multiplicacion de matrices va de afuera para adentro     
        
        vertices = mulMat(mulMat(mulMat(mulMat(self.Viewport,self.Projection), self.View), self.Model), aumentado)
        vf = Vector3(
            round(vertices[0][0]/vertices[3][0]),
            round(vertices[1][0]/vertices[3][0]),
            round(vertices[2][0]/vertices[3][0])
        )
        return vf

#el rotate tiene los angulos medidos en radianes
    def loadModelMatrix(self, translate, scale, rotate):
        translate = Vector3(*translate)
        rotate = Vector3(*rotate)
        scale = Vector3(*scale)
        translate_matrix = [
            [1.0,0.0,0.0,translate.x],
            [0.0,1.0,0.0,translate.y],
            [0.0,0.0,1.0,translate.z],
            [0.0,0.0,0.0,1.0],
        ]
        scale_matrix = [
            [scale.x,0.0,0.0,0.0],
            [0.0,scale.y,0.0,0.0],
            [0.0,0.0,scale.z,0.0],
            [0.0,0.0,0.0,1.0],
        ]

        rotation_matrix_x = [
            [1.0,0.0,0.0,0.0],
            [0.0,cos(rotate.x),-sin(rotate.x),0.0],
            [0.0,sin(rotate.x), cos(rotate.x),0.0],
            [0.0,0.0,0.0,1.0]
        ]
        rotation_matrix_y = [
            [cos(rotate.y),0.0,sin(rotate.y),0.0],
            [0.0,1.0,0.0,0.0],
            [-sin(rotate.y),0.0, cos(rotate.y),0.0],
            [0.0,0.0,0.0,1.0]
        ]
        rotation_matrix_z = [
            [cos(rotate.z),-sin(rotate.z),0.0,0.0],
            [sin(rotate.z), cos(rotate.z),0.0,0.0],
            [0.0,0.0,1.0,0.0],
            [0.0,0.0,0.0,1.0]
        ]
        
        rotation_matrix = mulMat(mulMat(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
        self.Model = mulMat(mulMat(translate_matrix, rotation_matrix), scale_matrix)
    
    #eye: ubicacion xyz de la camara 
    #center: punto al que la camara esta viendo
    #up: vector que nos dice que es arriba para la camara (vector u)
    def lookAt(self, eye, up, center):
        
        #z es el vector mas facil de obtener, es el vector que va del centro al ojo
        z = normalizar(restaVectorial(eye,center))
        x = normalizar(prodCruz(up,z))
        y = normalizar(prodCruz(z,x))

        self.loadViewMatrix(x,y,z, center)
        self.loadProyectionMatrix(-1.0/magnitud(restaVectorial(eye,center)))
        
    def loadViewportMatrix(self, x=0, y=0):
        self.Viewport = [
            [self.width/2,0.0,0.0,y+(self.width/2)],
            [0.0,self.height/2,0.0,x+(self.height/2)],
            [0.0,0.0,128.0,128.0],
            [0.0,0.0,0.0,1.0],
        ]

#se carga la matriz de proyeccion
    def loadProyectionMatrix(self, coeff):
        self.Projection = [
            [1.0,0.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
            [0.0,0.0,1.0,0.0],
            [0.0,0.0,-coeff,1.0],
        ]
# se carga la matriz de view
    def loadViewMatrix(self, x, y, z, center):
        M = [
            [x.x, x.y, x.z,0.0],
            [y.x, y.y, y.z,0.0],
            [z.x, z.y, z.z,0.0],
            [0.0,0.0,0.0,1.0]

        ]
        O_ = [
            [1.0,0.0,0.0,-center.x],
            [0.0,1.0,0.0,-center.y],
            [0.0,0.0,1.0,-center.z],
            [0.0,0.0,0.0,1.0]

        ]
        self.View = mulMat(M, O_)


def gouradPlanet(render, x, y, **kwargs):
    w,v,u = kwargs["bar"]
    nA, nB, nC = kwargs["normales"]

    luz = kwargs["light"]
    normx = nA.x*w + nB.x*v + nC.x*u 
    normy = nA.y*w + nB.y*v + nC.y*u 
    normz = nA.z*w + nB.z*v + nC.z*u 
    vnormal = Vector3(normx, normy, normz)
    intensity = prodPunto(vnormal, luz)
    if intensity < 0:
        intensity =0
    elif intensity >1:
        intensity =1
    
    #lista de colores a utilizar
    sand_grey = (146,139,133)
    wolf_grey = (156,156,154)
    light_brown = (112,96,80)
    wood = (123,82,52)
    white = (255, 255, 255)
    dark_grey = (135,135,123)
    dark_wood = (58,53,50)
    browngrey = (152,148,137)

    pnoise = p.Perlin()
    for m in range(800):
        for n in range(800):
            col = [int((pnoise.value(x/900.0, y/10.0, 0)+1) *150), ] *3

            if col[0] >=220 and col[0] <=200:
                mul = [int((browngrey[0]*col[0]/255.0)* intensity),int((browngrey[1]*col[1]/255.0)* intensity),
                    int((browngrey[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])

            if col[0] > 250:
                mul = [int((light_brown[0]*col[0]/255.0)* intensity),int((light_brown[1]*col[1]/255.0)* intensity),
                int((light_brown[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])

            if col[0] >190: 
                mul = [int((wolf_grey[0]*col[0]/255.0)* intensity),int((wolf_grey[1]*col[1]/255.0)* intensity),
                    int((wolf_grey[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])
            
            if col[0] >=120 and col[0] <=160:
                mul = [int((dark_wood[0]*col[0]/255.0)* intensity),int((dark_wood[1]*col[1]/255.0)* intensity),
                    int((dark_wood[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])

            if col[0] >170: 
                mul = [int((wood[0]*col[0]/255.0)* intensity),int((wood[1]*col[1]/255.0)* intensity),
                    int((wood[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])
            
            if col[0] >100: 
                mul = [int((sand_grey[0]*col[0]/255.0)* intensity),int((sand_grey[1]*col[1]/255.0)* intensity),
                    int((sand_grey[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])
            else:
                mul = [int((white[0]*col[0]/255.0)* intensity),int((white[1]*col[1]/255.0)* intensity),
                    int((white[2]*col[2]/255.0)* intensity)]
                if mul[0] < 0: mul[0] =0
                if mul[1] < 0: mul[1] = 0
                if mul[2] < 0: mul[2] = 0
                return color(mul[0], mul[1], mul[2])



def Estrellas():
    r = bm
    glClearColor(0,0,0)    
    for x in range(1, r.width-1):
        for y in range(1, r.height-1):
            st = ri(0, 1000)
            tamano = ri(0,2) #determinara el tamano de las estrellas
            try:
                if (st  == 3):
                    if tamano ==0: # estrella de tamaño de 1 pixeles
                        r.framebuffer[y][x] = color(255,255,255)
                    if tamano ==1: # estrella de tamaño de 4 pixeles
                        r.framebuffer[y][x] = color(255,255,255)
                        r.framebuffer[y][x+1] = color(255,255,255)
                        r.framebuffer[y+1][x] = color(255,255,255)
                        r.framebuffer[y+1][x+1] = color(255,255,255)
                    if tamano ==2: # estrella de tamaño de 9 pixeles
                        r.framebuffer[y][x] = color(255,255,255)
                        r.framebuffer[y][x+1] = color(255,255,255)
                        r.framebuffer[y][x+2] = color(255,255,255)
                        r.framebuffer[y+1][x] = color(255,255,255)
                        r.framebuffer[y+1][x+1] = color(255,255,255)
                        r.framebuffer[y+1][x+2] = color(255,255,255)
                        r.framebuffer[y+2][x] = color(255,255,255)
                        r.framebuffer[y+2][x+1] = color(255,255,255)
                        r.framebuffer[y+2][x+2] = color(255,255,255)
            except IndexError:
                pass