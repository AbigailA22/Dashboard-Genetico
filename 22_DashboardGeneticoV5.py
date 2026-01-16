import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import random
import numpy as np
import math

def compute_overlap_area(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx > 0 and dy > 0:
        return dx * dy
    return 0

def fitnessFunction(rectangles, container_width, container_height, margen=1):
    container_area = container_width * container_height
    overlap_penalty = 0
    margin_penalty = 0
    resolution = 2
    grid = np.zeros((container_width//resolution, container_height//resolution)) #aproxima área cubierta

    for i, rect in enumerate(rectangles):
        x, y, w, h = rect
        x_end, y_end = min(container_width, x+w), min(container_height, y+h)
        grid[int(x//resolution):int(x_end//resolution),
     int(y//resolution):int(y_end//resolution)] = 1

        for j in range(i+1, len(rectangles)):
            overlap_penalty += compute_overlap_area(rect, rectangles[j])

    covered_area = np.sum(grid) * (resolution**2)
    fill_ratio = covered_area / container_area

    for i, rect in enumerate(rectangles):
        for j in range(i+1, len(rectangles)):
            x1,y1,w1,h1 = rect
            x2,y2,w2,h2 = rectangles[j]
            dx = max(0, max(x2-(x1+w1), x1-(x2+w2)))
            dy = max(0, max(y2-(y1+h1), y1-(y2+h2)))
            min_dist = math.sqrt(dx**2 + dy**2)
            if min_dist < margen:
                margin_penalty += (margen-min_dist)*50

    return fill_ratio*1000 - overlap_penalty*10 - margin_penalty

def clip_rectangles(individuo, contenedorAncho, contenedorAlto):
    rects_clipped = []
    for (x, y, w, h) in individuo:
        w = max(10, min(w, contenedorAncho))
        h = max(10, min(h, contenedorAlto))
        x = max(0, min(x, contenedorAncho - w))
        y = max(0, min(y, contenedorAlto - h))
        rects_clipped.append((x, y, w, h))
    return rects_clipped

def blx_alpha_crossover(p1, p2, alpha=0.5):
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    diff = hi - lo
    child = np.random.uniform(lo - alpha*diff, hi + alpha*diff)
    return child

def adaptive_mutation(x, gen, num_generacion, bounds, numRectangulos):
    sigma = 0.5 * (1 - gen/num_generacion)  
    mutated = x + np.random.normal(0, sigma, size=x.shape)
    min_bounds = np.tile([b[0] for b in bounds], numRectangulos)
    max_bounds = np.tile([b[1] for b in bounds], numRectangulos)  
    return np.clip(mutated, min_bounds, max_bounds)

def binary_tournament_selection(poblacion, fitness, k=3):
    padres = []
    for _ in range(2): 
        candidatos = random.sample(range(len(poblacion)), k)
        mejor = max(candidatos, key=lambda i: fitness[i])
        padres.append(poblacion[mejor])
    return padres

def valoresRectangulo(contenedorAncho, contenedorAlto):
    w = random.randint(contenedorAncho // 5, contenedorAncho // 2)
    h = random.randint(contenedorAlto // 5, contenedorAlto // 2)
    x = random.randint(0, contenedorAncho - w)
    y = random.randint(0, contenedorAlto - h)
    return (x, y, w, h)

def run_experiment(numRectangulos, contenedorAncho, contenedorAlto, margen,
                   ejecucionesIndependientes, sizePoblacion, numGen, pc, pm):
    poblacion = [[valoresRectangulo(contenedorAncho, contenedorAlto) 
                  for _ in range(numRectangulos)] for _ in range(sizePoblacion)]
    
    bounds = [(0, contenedorAncho), (0, contenedorAlto), 
              (20, contenedorAncho//2), (20, contenedorAlto//2)]
    mejor_individuo, mejor_fitness = None, -float("inf")

    convergence_stats = []

    for ejecucion in range(ejecucionesIndependientes):       
        for gen in range(numGen):
            fitness = [fitnessFunction(ind, contenedorAncho, contenedorAlto, margen) for ind in poblacion]

            convergence_stats.append({
                "gen": gen,
                "best": np.max(fitness),
                "worst": np.min(fitness),
                "mean": np.mean(fitness)
            })

            new_population = []
            while len(new_population) < sizePoblacion:
                parents = binary_tournament_selection(poblacion, fitness, k=3)

                if random.uniform(0, 100) <= pc:
                    child = blx_alpha_crossover(np.array(parents[0]).flatten(),
                                                np.array(parents[1]).flatten())
                    child = child.reshape(numRectangulos, 4)
                else:
                    child = np.copy(parents[0])

                if random.uniform(0, 100) <= pm:
                    child = adaptive_mutation(np.array(child).flatten(), gen, numGen, bounds, numRectangulos)
                    child = child.reshape(numRectangulos, 4)

                child_tuplas = [tuple(r) for r in child.tolist()]
                child_tuplas = clip_rectangles(child_tuplas, contenedorAncho, contenedorAlto)
                new_population.append(child_tuplas)

            poblacion_total = poblacion + new_population
            fitness_total = [fitnessFunction(ind, contenedorAncho, contenedorAlto, margen) for ind in poblacion_total]
            indices_ordenados = np.argsort(fitness_total)[::-1]
            poblacion = [poblacion_total[i] for i in indices_ordenados[:sizePoblacion]]

            if fitness_total[indices_ordenados[0]] > mejor_fitness:
                mejor_fitness = fitness_total[indices_ordenados[0]]
                mejor_individuo = poblacion[0]

    return mejor_individuo, mejor_fitness, convergence_stats


#tkinter part
def plot_solution(ax, rectangulos, contenedorAncho, contenedorAlto, aptitud, title):
    ax.clear()
    for x, y, width, height in rectangulos:
        rect = patches.Rectangle((x, y), width, height,
                                 linewidth=1, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect)
    ax.set_xlim(0, contenedorAncho)
    ax.set_ylim(0, contenedorAlto)
    ax.set_aspect('equal')
    ax.set_title(f"{title} - Fitness = {aptitud:.2f}")
    ax.grid(True)

def plot_convergence(ax, stats):
    ax.clear()
    gens = [s["gen"] for s in stats]
    best = [s["best"] for s in stats]
    worst = [s["worst"] for s in stats]
    mean = [s["mean"] for s in stats]

    ax.plot(gens, best, label="Mejor", color="green")
    ax.plot(gens, worst, label="Peor", color="red")
    ax.plot(gens, mean, label="Promedio", color="blue")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.set_title("Convergencia Evolutiva")
    ax.legend()
    ax.grid(True)

def run_and_show(numRectangulos, margen):
    numRectangulos = int(numRectangulos_var.get())
    margen = int(margen_var.get())
    
    contenedorAncho, contenedorAlto = 100, 100
    ejecucionesIndependientes = 1  
    sizePoblacion = 100
    numGen = 200
    pc = 90
    pm = 20

    #Primera solución 
    poblacion_inicial = [[valoresRectangulo(contenedorAncho, contenedorAlto)
                          for _ in range(numRectangulos)]]
    aptitud_inicial = fitnessFunction(poblacion_inicial[0], contenedorAncho, contenedorAlto, margen)

    #Solución final y estadísticas
    rectangulos_final, aptitud_final, stats = run_experiment(numRectangulos, contenedorAncho,
                                                            contenedorAlto, margen,
                                                            ejecucionesIndependientes,
                                                            sizePoblacion, numGen, pc, pm)

    plot_solution(ax1, poblacion_inicial[0], contenedorAncho, contenedorAlto, aptitud_inicial, "Primera solución")
    plot_solution(ax2, rectangulos_final, contenedorAncho, contenedorAlto, aptitud_final, "Última solución")
    plot_convergence(ax3, stats)

    canvas1.draw()
    canvas2.draw()
    canvas3.draw()

root = tk.Tk()
root.title("Interfaz - Dashboard Genético")

frame = ttk.Frame(root)
frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

#Parámetro numero Rectangulos
ttk.Label(frame, text="Número de Gráficos:").grid(row=0, column=0, sticky="w")
numRectangulos_var = tk.IntVar(value=6)
ttk.Entry(frame, textvariable=numRectangulos_var, width=5).grid(row=0, column=1)

#Parámetro margen
ttk.Label(frame, text="Margen mínimo:").grid(row=1, column=0, sticky="w")
margen_var = tk.IntVar(value=2)
ttk.Entry(frame, textvariable=margen_var, width=5).grid(row=1, column=1)

#Botón ejecutar
ttk.Button(frame, text="Ejecutar", command=lambda: run_and_show(int(numRectangulos_var.get()), int(margen_var.get()) )).grid(row=2, column=0, columnspan=2, pady=5)

#Parte gráfica
fig1, ax1 = plt.subplots(figsize=(4,4))
fig2, ax2 = plt.subplots(figsize=(4,4))

canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig3, ax3 = plt.subplots(figsize=(5,4))
canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

root.mainloop()
