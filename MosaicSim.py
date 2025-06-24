import numpy as np
import cv2
import networkx as nx
import generativepy.color
import generativepy.geometry
from generativepy.color import Color
from generativepy.drawing import make_image, setup
from scipy.spatial import Voronoi
import math
import random as rd
import jax.numpy as jnp
import jax
import pandas as pd
import jaxopt
import os

jax.config.update("jax_enable_x64", True)

DC_RADIUS = 0.6
DC_DISTANCE = 0.6

class Cone:
    def __init__(self, color, pos, is_double, id):
        self.color = color
        self.orig_pos = pos
        self.pos = pos
        self.is_double = is_double
        self.id=id
        self.doublable = self.color != 'b'

class DoubleCone:
    def __init__(self, cone1, cone2, centre):
            self.cone1 = cone1
            self.cone2 = cone2
            self.centre = centre

    def update_centre(self):
        self.centre = (self.cone1.pos+self.cone2.pos)/2

class Mosaic:

    def __init__(self, delta, sigma, mu, tau):
        self.cones = []
        self.sc = []
        self.dc = []
        self.width = None
        self.height = None
        self.delta = delta
        self.sigma = sigma
        self.mu = mu
        self.tau = tau

    def init_from_csv(self, src, img_src, compress=1):
        df = pd.read_csv(src)

        image = cv2.imread(img_src, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50)
        circles = np.round(circles[0, :]).astype("int")
        mean_r = sum([circle[2] for circle in circles])/len(circles)
        self.width = image.shape[1] * 0.5/mean_r * compress + 5
        self.height = image.shape[0] * 0.5/mean_r * compress + 5

        x = None
        y = None
        color = None
        cone_id = None
        id_to_cone = dict()

        for col in df:
            if "Cone" in col:
                axis = col.split(" ")[2]
                if axis == 'x':
                    x = df[col][len(df)-1]
                    color = col.split(" ")[3][1]
                    cone_id = int(col.split(" ")[1])
                else:
                    y = df[col][len(df)-1]
                    cone = Cone(color, np.array([x, y]), False, cone_id)
                    self.cones.append(cone)
                    id_to_cone[cone_id] = cone
        dcs = eval(f"{df['Doubles'][len(df)-1]}")
        for c1, c2 in dcs:
            id_to_cone[c1].is_double = True
            id_to_cone[c2].is_double = True
            self.dc.append(DoubleCone(id_to_cone[c1], id_to_cone[c2], (id_to_cone[c1].pos+id_to_cone[c2].pos)/2))
        for cone in self.cones:
            if not cone.is_double:
                self.sc.append(cone)

            
    def init_from_image(self, src, compress=1):  
        
        image = cv2.imread(src, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50)
        circles = np.round(circles[0, :]).astype("int")
        mean_r = sum([circle[2] for circle in circles])/len(circles)
        self.width = image.shape[1] * 0.5/mean_r * compress + 5
        self.height = image.shape[0] * 0.5/mean_r * compress + 5

        count = 0

        for (x, y, r) in circles:
            bgr = image[y-1][x-1]
            if max(bgr) == bgr[0]:
                cone = Cone('b', np.array([x*0.5/mean_r*compress+2.5, y*0.5/mean_r*compress+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            elif max(bgr) == bgr[1]:
                cone = Cone('g', np.array([x*0.5/mean_r*compress+2.5, y*0.5/mean_r*compress+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            else:
                cone = Cone('r', np.array([x*0.5/mean_r*compress+2.5, y*0.5/mean_r*compress+2.5]), False, count)
                self.cones.append(cone)
                self.sc.append(cone)
            count += 1
        
        G = nx.Graph()
        G.add_nodes_from(self.cones)
        for i in range(len(self.cones)):
            for j in range(i+1, len(self.cones)):
                if not (self.cones[i].color == 'b' and self.cones[j].color == 'b') and np.linalg.norm(self.cones[i].orig_pos - self.cones[j].orig_pos) < self.tau:
                    G.add_edge(self.cones[i], self.cones[j])
        blues = [cone for cone in self.sc if cone.color == 'b']
        non_doublables = list(set(nx.maximal_independent_set(G, nodes=blues)) - set(blues))
        while len(non_doublables) < max(round((len(self.sc)/5-len(blues))), 0):
            non_doublables = list(set(nx.maximal_independent_set(G, nodes=blues)) - set(blues))

        for cone in non_doublables[:max(round((len(self.sc)/5-len(blues))), 0)]:
            cone.doublable = False

    def make_doubles(self, radius):
        doublables = [cone for cone in self.sc if cone.doublable and np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) <= radius]
        edges = []
        for i in range(len(doublables)):
                for j in range(i+1, len(doublables)):
                    if np.linalg.norm(np.array([max(abs(doublables[i].orig_pos[0]-doublables[j].orig_pos[0])-2*self.delta, 0),max(abs(doublables[i].orig_pos[1]-doublables[j].orig_pos[1])-2*self.delta, 0)])) <= DC_DISTANCE:
                        edges.append((doublables[i], doublables[j]))
        matching = []
        used_nodes = set()
        rd.shuffle(edges)
        for edge in edges:
            if edge[0] not in used_nodes and edge[1] not in used_nodes:
                matching.append(edge)
                used_nodes.update(edge)
        for cone1, cone2 in matching:
            cone1.is_double = True
            cone2.is_double = True
            self.sc.remove(cone1)
            self.sc.remove(cone2)
            cone1_to_cone2 = cone2.orig_pos - cone1.orig_pos
            centre = cone1_to_cone2/2 + cone1.orig_pos
            cone2.pos = centre + (cone1_to_cone2/np.linalg.norm(cone1_to_cone2)) * DC_DISTANCE/2
            cone1.pos = centre + -1*(cone1_to_cone2/np.linalg.norm(cone1_to_cone2)) * DC_DISTANCE/2
            self.dc.append(DoubleCone(cone1, cone2, centre))

    def optimize(self, radius):

        sings = jnp.array([1.0 if not s.doublable and np.linalg.norm(s.orig_pos - np.array([self.width/2, self.height/2])) <= radius else 0.0 for s in self.sc])
        sing_mask = jnp.outer(sings, sings)
        sing_mask *= jnp.triu(jnp.ones_like(sing_mask), 1)
        mosaic_mask = jnp.outer(sings, jnp.ones(len(self.dc)))

        sing_dc_close_mask = jnp.array([[jnp.linalg.norm(jnp.array([max(abs(self.sc[i].orig_pos[0]-(self.dc[j].cone1.orig_pos[0]+self.dc[j].cone2.orig_pos[0])/2)-2*self.delta, 0),max(abs(self.sc[i].orig_pos[1]-(self.dc[j].cone1.orig_pos[1]+self.dc[j].cone2.orig_pos[1])/2)-2*self.delta, 0)])) <= DC_DISTANCE/2 + DC_RADIUS + 0.5 for j in range(len(self.dc))] for i in range(len(self.sc))])
        sing_close_mask = jnp.array([[jnp.linalg.norm(jnp.array([max(abs(self.sc[i].orig_pos[0]-self.sc[j].orig_pos[0])-2*self.delta, 0),max(abs(self.sc[i].orig_pos[1]-self.sc[j].orig_pos[1])-2*self.delta, 0)])) <= 1 for j in range(len(self.sc))] for i in range(len(self.sc))])
        dc_close_mask = jnp.array([[jnp.linalg.norm(jnp.array([max(abs((self.dc[i].cone1.orig_pos[0]+self.dc[i].cone2.orig_pos[0])/2-(self.dc[j].cone1.orig_pos[0]+self.dc[j].cone2.orig_pos[0])/2)-2*self.delta, 0),max(abs((self.dc[i].cone1.orig_pos[1]+self.dc[i].cone2.orig_pos[1])/2-(self.dc[j].cone1.orig_pos[1]+self.dc[j].cone2.orig_pos[1])/2)-2*self.delta, 0)])) <= DC_DISTANCE + 2*DC_RADIUS for j in range(len(self.dc))] for i in range(len(self.dc))])

        sing_dc_int_mask = jnp.array([[jnp.linalg.norm(jnp.array([max(abs(self.sc[i].orig_pos[0]-(self.dc[j].cone1.orig_pos[0]+self.dc[j].cone2.orig_pos[0])/2)-2*self.delta, 0),max(abs(self.sc[i].orig_pos[1]-(self.dc[j].cone1.orig_pos[1]+self.dc[j].cone2.orig_pos[1])/2)-2*self.delta, 0)])) <= 0 for j in range(len(self.dc))] for i in range(len(self.sc))])

        sing_space_mask = jnp.array([[jnp.linalg.norm(jnp.array([max(abs(self.sc[i].orig_pos[0]-self.sc[j].orig_pos[0])-2*self.delta, 0),max(abs(self.sc[i].orig_pos[1]-self.sc[j].orig_pos[1])-2*self.delta, 0)])) <= 2*DC_RADIUS for j in range(len(self.sc))] for i in range(len(self.sc))])

        def repl(mat, rad1, rad2):
            space = rad1 + rad2
            return 10*(jnp.sqrt(mat)-(space)*jnp.ones_like(mat))**2 * jnp.heaviside((space)**2*jnp.ones_like(mat)-mat, 0)
        
        def sing_space(mat, space):
            return 10*jnp.exp(-30*(jnp.sqrt(mat)+(0.2-space)*jnp.ones_like(mat)))
        
        def corr_mos(mat, rad1, rad2):
            a = 1/5
            b = 15
            return -a*jnp.ones_like(mat)/((b*(jnp.sqrt(mat)-(rad1+rad2)*jnp.ones_like(mat)))**2 + jnp.ones_like(mat))
        
        def loss(x):
            sing_x = x[0:2*len(self.sc):2]
            sing_y = x[1:2*len(self.sc):2]
            cent_x = x[2*len(self.sc):2*(len(self.sc)+len(self.dc)):2]
            cent_y = x[2*len(self.sc)+1:2*(len(self.sc)+len(self.dc)):2]
            thetas = x[2*(len(self.sc)+len(self.dc)):]

            cone1_x = cent_x + DC_DISTANCE/2 * jnp.cos(thetas)
            cone1_y = cent_y + DC_DISTANCE/2 * jnp.sin(thetas)
            cone2_x = cent_x - DC_DISTANCE/2 * jnp.cos(thetas)
            cone2_y = cent_y - DC_DISTANCE/2 * jnp.sin(thetas)

            sing_d2 = (jnp.outer(sing_x, jnp.ones_like(sing_x)) - jnp.outer(jnp.ones_like(sing_x), sing_x))**2 + (jnp.outer(sing_y, jnp.ones_like(sing_y)) - jnp.outer(jnp.ones_like(sing_y), sing_y))**2
            c1_sing_d2 = (jnp.outer(sing_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(sing_x), cone1_x))**2 + (jnp.outer(sing_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(sing_y), cone1_y))**2
            c2_sing_d2 = (jnp.outer(sing_x, jnp.ones_like(cone2_x)) - jnp.outer(jnp.ones_like(sing_x), cone2_x))**2 + (jnp.outer(sing_y, jnp.ones_like(cone2_y)) - jnp.outer(jnp.ones_like(sing_y), cone2_y))**2
            c1_c2_d2 = (jnp.outer(cone1_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(cone2_x), cone2_x))**2 + (jnp.outer(cone1_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(cone2_y), cone2_y))**2
            c1_c1_d2 = (jnp.outer(cone1_x, jnp.ones_like(cone1_x)) - jnp.outer(jnp.ones_like(cone1_x), cone1_x))**2 + (jnp.outer(cone1_y, jnp.ones_like(cone1_y)) - jnp.outer(jnp.ones_like(cone1_y), cone1_y))**2
            c2_c2_d2 = (jnp.outer(cone2_x, jnp.ones_like(cone2_x)) - jnp.outer(jnp.ones_like(cone2_x), cone2_x))**2 + (jnp.outer(cone2_y, jnp.ones_like(cone2_y)) - jnp.outer(jnp.ones_like(cone2_y), cone2_y))**2

            sing_repl = jnp.sum(repl(sing_d2+jnp.identity(sing_d2.shape[0]), 0.5, 0.5) * sing_close_mask * jnp.triu(jnp.ones_like(sing_d2), 1))
            c1_sing_repl = jnp.sum(repl(c1_sing_d2, 0.5, DC_RADIUS) * sing_dc_close_mask)
            c2_sing_repl = jnp.sum(repl(c2_sing_d2, 0.5, DC_RADIUS) * sing_dc_close_mask)
            c1_c2_repl = jnp.sum(repl(c1_c2_d2, DC_RADIUS, DC_RADIUS) * dc_close_mask * (jnp.triu(jnp.ones_like(c1_c2_d2), 1) + jnp.tril(jnp.ones_like(c1_c2_d2), -1)))
            c1_c1_repl = jnp.sum(repl(c1_c1_d2+jnp.identity(c1_c1_d2.shape[0]), DC_RADIUS, DC_RADIUS) * dc_close_mask * (jnp.triu(jnp.ones_like(c1_c1_d2), 1)))
            c2_c2_repl = jnp.sum(repl(c2_c2_d2+jnp.identity(c2_c2_d2.shape[0]), DC_RADIUS, DC_RADIUS) * dc_close_mask * (jnp.triu(jnp.ones_like(c2_c2_d2), 1)))

            sing_sep = jnp.sum(sing_space(sing_d2+jnp.identity(sing_d2.shape[0]), self.mu) * sing_mask * sing_space_mask)

            mosaic_c1 = jnp.sum(corr_mos(c1_sing_d2, 1.2, 0) * mosaic_mask * sing_dc_int_mask)
            mosaic_c2 = jnp.sum(corr_mos(c2_sing_d2, 1.2, 0) * mosaic_mask * sing_dc_int_mask)

            return sing_repl + c1_sing_repl + c2_sing_repl + c1_c2_repl + c1_c1_repl + c2_c2_repl + mosaic_c1 + mosaic_c2 + sing_sep
        
        x0 = []
        lbound = []
        ubound = []

        for cone in self.sc:
            x0.append(cone.pos[0])
            x0.append(cone.pos[1])
            lbound.append(cone.orig_pos[0]-self.delta)
            ubound.append(cone.orig_pos[0]+self.delta)
            lbound.append(cone.orig_pos[1]-self.delta)
            ubound.append(cone.orig_pos[1]+self.delta)
        for dc in self.dc:
            x0.append(dc.centre[0])
            x0.append(dc.centre[1])
            lbound.append((dc.cone1.orig_pos[0]+dc.cone2.orig_pos[0])/2-self.delta)
            ubound.append((dc.cone1.orig_pos[0]+dc.cone2.orig_pos[0])/2+self.delta)
            lbound.append((dc.cone1.orig_pos[1]+dc.cone2.orig_pos[1])/2-self.delta)
            ubound.append((dc.cone1.orig_pos[1]+dc.cone2.orig_pos[1])/2+self.delta)
        for dc in self.dc:
            ang = np.arctan2(dc.cone1.pos[1]-dc.centre[1], dc.cone1.pos[0]-dc.centre[0])
            x0.append(ang)
            lbound.append(-jnp.inf)
            ubound.append(jnp.inf)

        optimizer = jaxopt.LBFGSB(fun=loss, maxiter=10000, verbose=False, tol=0.001)
        res = optimizer.run(init_params=jnp.array(x0), bounds=(jnp.array(lbound), jnp.array(ubound)))
        sc_pos = [np.array([res.params[2*i], res.params[2*i+1]]) for i in range(len(self.sc))]
        dc_centre = [np.array([res.params[2*(len(self.sc)+i)], res.params[2*(len(self.sc)+i)+1]]) for i in range(len(self.dc))]
        dc_cone1_pos = [dc_centre[i] + DC_DISTANCE/2 * np.array([math.cos(res.params[2*(len(self.sc)+len(self.dc))+i]), math.sin(res.params[2*(len(self.sc)+len(self.dc))+i])]) for i in range(len(self.dc))]
        dc_cone2_pos = [dc_centre[i] - DC_DISTANCE/2 * np.array([math.cos(res.params[2*(len(self.sc)+len(self.dc))+i]), math.sin(res.params[2*(len(self.sc)+len(self.dc))+i])]) for i in range(len(self.dc))]
            
        for i in range(len(self.sc)):
            self.sc[i].pos = sc_pos[i]
        for i in range(len(self.dc)):
            self.dc[i].centre = dc_centre[i]
            self.dc[i].cone1.pos = dc_cone1_pos[i]
            self.dc[i].cone2.pos = dc_cone2_pos[i]

        return res.state.error <= 0.001


    def get_next_row(self):
        row = [self.rate_mos(edge_threshold=2)]
        for cone in self.cones:
            row.append(cone.pos[0])
            row.append(cone.pos[1])
        doubles = [(dc.cone1.id, dc.cone2.id) for dc in self.dc]
        row.append(doubles)
        return row
    
    def execute(self, dir):
        os.makedirs(dir, exist_ok=True)
        cols = ['Margin 2 Mosaic Score']
        for cone in self.cones:
            cols.append(f'Cone {cone.id} x ({cone.color})')
            cols.append(f'Cone {cone.id} y ({cone.color})')
        cols.append('Doubles')
        df = pd.DataFrame(columns=cols)
        df.loc[len(df)] = self.get_next_row()
        self.export(f'{dir}/0.png')
        num_iter = math.ceil(max([np.linalg.norm(cone.orig_pos - np.array([self.width/2, self.height/2])) for cone in self.sc])/self.sigma)
        for i in range(1, num_iter+1):
            self.make_doubles(i * self.sigma)
            if not self.optimize(i * self.sigma):
                fh = open(f'{dir}/error.txt', 'a')
                fh.write(f'FAILED at iteration {i}\n')
                fh.close()
            self.export(f'{dir}/{i}.png')
            df.loc[len(df)] = self.get_next_row()
            jax.clear_caches()
        df.to_csv(f'{dir}/res.csv', index=False)
    
    def export(self, filename):
        def draw(ctx, pixel_width, pixel_height, frame_no, frame_count):
            colordict = {'r': 'red', 'g': 'green', 'b': 'blue', 'u': 'purple'}
            setup(ctx, pixel_width, pixel_height, background=generativepy.color.Color(1))
            pointlist = [cone.pos for cone in self.cones] + [np.array([-self.width, -self.height]), np.array([2*self.width, -self.height]), np.array([-self.width, 2*self.height]), np.array([2*self.width, 2*self.height])]
            voronoi = Voronoi(pointlist)
            for i in range(len(self.cones)):
                vorpoints = [voronoi.vertices[index] for index in voronoi.regions[voronoi.point_region[i]]]
                generativepy.geometry.Polygon(ctx).of_points([50*(point) for point in vorpoints]).fill(pattern=generativepy.color.Color(colordict[self.cones[i].color])).stroke(pattern=generativepy.color.Color(0), line_width=1)
            for cone in self.cones:
                generativepy.geometry.Circle(ctx).of_center_radius(50*cone.pos, 2.5).fill(generativepy.color.Color(0))
                generativepy.geometry.Circle(ctx).of_center_radius(50*cone.pos, 50*(DC_RADIUS if cone.is_double else 0.5)).stroke(pattern=generativepy.color.Color(0, 0, 0, 0.3), line_width=1)
                generativepy.geometry.Text(ctx).of(str(cone.id), (50*cone.pos[0]+5,50*cone.pos[1]+5)).align_top().size(10).fill(Color(0))
            for cone in [(dc.cone1, dc.cone2) for dc in self.dc]:
                generativepy.geometry.Line(ctx).of_start_end(50*(cone[0].pos), 50*(cone[1].pos)).stroke(pattern=generativepy.color.Color(0), line_width=1)
                
        make_image(filename, draw, round(50*self.width), round(50*self.height))

    def rate_mos(self, edge_threshold=2):
        if len(self.dc) < 4:
            return 0
        count = 0
        min_x = min([cone.orig_pos[0] for cone in self.cones if not cone.doublable]) + edge_threshold
        max_x = max([cone.orig_pos[0] for cone in self.cones if not cone.doublable]) - edge_threshold
        min_y = min([cone.orig_pos[1] for cone in self.cones if not cone.doublable]) + edge_threshold
        max_y = max([cone.orig_pos[1] for cone in self.cones if not cone.doublable]) - edge_threshold
        non_doublables = [cone for cone in self.cones if not cone.doublable and cone.orig_pos[0] >= min_x and cone.orig_pos[0] <= max_x and cone.orig_pos[1] >= min_y and cone.orig_pos[1] <= max_y]
        for sc in non_doublables:
            is_valid = True
            dc_dists = [np.linalg.norm(dc.centre - sc.pos) for dc in self.dc]
            dc_enum = list(enumerate(dc_dists))
            dc_enum = sorted(dc_enum, key=lambda x: x[1])
            for i in range(4):
                is_valid &= dc_enum[i][1] >= math.sqrt(1.2**2 - (DC_DISTANCE/2)**2) - 0.2 and dc_enum[i][1] <= math.sqrt(1.2**2 - (DC_DISTANCE/2)**2) + 0.2
                sc_to_cent = self.dc[dc_enum[i][0]].centre - sc.pos
                cent_to_cone1 = self.dc[dc_enum[i][0]].cone1.pos - self.dc[dc_enum[i][0]].centre
                angle = math.acos(max(min(np.dot(sc_to_cent, cent_to_cone1)/(np.linalg.norm(sc_to_cent) * np.linalg.norm(cent_to_cone1)), 1.0), -1.0))
                is_valid &= angle >= math.pi/2-0.3 and angle <= math.pi/2+0.3
            if is_valid:
                count += 1
        return count/len(non_doublables)


if __name__ == '__main__':

    mosaic = Mosaic(2, 1, 2.3, 2)
    mosaic.init_from_image("Images/Sab.jpg", compress=1.1)
    mosaic.execute('Output')