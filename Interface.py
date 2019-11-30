from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab
import cv2
import numpy as np
import CycleGAN as cc

import tensorflow as tf

from math import inf

import matplotlib as plt
plt.use("TkAgg")

exitFlag = False

class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = "black"
        self.color_bg = "white"
        self.old_x = None
        self.old_y = None
        self.isPen = True
        self.penwidth = 3
        self.drawWidgets()
        self.c.bind("<B1-Motion>", self.paint)  # desenha a linha quando preciona o mouse
        self.c.bind("<ButtonRelease-1>", self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            if self.isPen:
                self.c.create_line(
                    self.old_x,
                    self.old_y,
                    e.x,
                    e.y,
                    width=self.penwidth,
                    fill=self.color_fg,
                    capstyle=tk.ROUND,
                    smooth=True,
                )
            else:
                self.c.create_line(
                    self.old_x,
                    self.old_y,
                    e.x,
                    e.y,
                    width=self.penwidth,
                    fill=self.color_bg,
                    capstyle=tk.ROUND,
                    smooth=True,
                )

        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # limpa o canvas
        self.old_x = None
        self.old_y = None

    def changeW(self, e):  #muda o tamanho da caneta
        self.penwidth = e

    def clear(self):
        self.c.delete(tk.ALL)

    def change_fg(self):  # muda a cor da caneta
        self.color_fg = tk.colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  # muda a cor do plano de fundo
        self.color_bg = tk.colorchooser.askcolor(color=self.color_bg)[1]
        self.c["bg"] = self.color_bg

    def selectPen(self):
        self.isPen = True

    def selectEraser(self):
        self.isPen = False
        
    def make_square(self, im, sqr_size, fill_color=[0, 0, 0]):
        y, x = im.shape[0:2]
        if y > sqr_size or x > sqr_size:
            if y > x:
                div = sqr_size / y
                y *= div
                x *= div
            else:
                div = sqr_size / x
                y *= div
                x *= div
        size = max(sqr_size, x, y)
        add_x = int((size - x)/2)
        add_y = int((size - y)/2)
        im = cv2.resize(im, (int(x), int(y)))
        new_im = cv2.copyMakeBorder(
            im, add_y, add_y, add_x, add_x, cv2.BORDER_CONSTANT, value=fill_color)
        return new_im

    def uploadImg(self):
        fileImg = tk.filedialog.askopenfilename()
        img = cv2.imread(fileImg)
        img = self.make_square(img, 256)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        self.c.image = ImageTk.PhotoImage(img_pil)
        self.c.create_image(0, 0, image=self.c.image, anchor='nw')

    def nextStep(self):
        x = root.winfo_rootx() + self.c.winfo_x()
        y = root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        pilImg = ImageGrab.grab().crop((x, y, x1, y1)).resize((256, 256))
        self.npImg = np.uint8(np.clip(pilImg, 0, 255))
        self.controls.destroy()
        self.menu.destroy()
        self.c.destroy()
        self.widgetsNextStep()

    def drawWidgets(self):
        self.controls = tk.Frame(self.master, padx=5, pady=5)

        self.butUp = ttk.Button(self.controls)
        self.butUp["text"] = "Subir Imagem"
        self.butUp["width"] = 15
        self.butUp["command"] = self.uploadImg
        self.butUp.grid(row=1, column=0)

        self.butNext = ttk.Button(self.controls)
        self.butNext["text"] = "Continuar"
        self.butNext["width"] = 15
        self.butNext["command"] = self.nextStep
        self.butNext.grid(row=1, column=1)

        tk.Label(self.controls, text="Tamanho:", font=(
            "arial 18")).grid(row=0, column=0)
        self.slider = ttk.Scale(
            self.controls, from_=1, to=10, command=self.changeW, orient=tk.HORIZONTAL
        )
        self.slider.set(self.penwidth)
        self.slider.grid(row=0, column=1, ipadx=30)
        self.controls.pack(side=tk.LEFT)

        self.c = tk.Canvas(self.master, width=256, height=256, bg=self.color_bg,)
        self.c.pack(fill=tk.BOTH, expand=True)

        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)
        colormenu = tk.Menu(self.menu)
        optionmenu = tk.Menu(self.menu)
        toolmenu = tk.Menu(self.menu)

        self.menu.add_cascade(label="Cores", menu=colormenu)
        colormenu.add_command(label="Cor da Caneta", command=self.change_fg)
        colormenu.add_command(label="Cor do Background",
                              command=self.change_bg)

        self.menu.add_cascade(label="Opcões", menu=optionmenu)
        optionmenu.add_command(label="Limpar Canvas", command=self.clear)
        optionmenu.add_command(label="Sair", command=self.master.destroy)

        self.menu.add_cascade(label="Ferramentas", menu=toolmenu)
        toolmenu.add_command(label="Selecionar Caneta", command=self.selectPen)
        toolmenu.add_command(label="Selecionar Borracha",
                             command=self.selectEraser)

        self.cycle = cc.CycleGAN(True)

    
    def NextGen(self):
        points = []
        for i in range(10):
            points.append(int(self.imgs[("photo_"+str(i))]["points"]["text"]))
        better = np.argmax(points)

        matingPool = []
        for i in range(10):
            if i != better:
                for _ in range((int(int(self.imgs[("photo_"+str(i))]["points"]["text"]) / int(self.imgs[("photo_"+str(better))]["points"]["text"]) * 1000)) + 1):
                    matingPool.append(i)

        matingPool = np.array(matingPool)

        plt.pyplot.close("all")

        for r in range(4):
            for c in range(5):
                lenDNA = len(self.imgs[("photo_"+str(better))]["DNA"])
                for w in range(lenDNA):
                    ww = []

                    crossPoint = np.random.randint(1, lenDNA-1)
                    if np.random.rand() >= 0.5:
                        for f in self.imgs[("photo_"+str(better))]["DNA"][:crossPoint]:
                            ww.append(f)
                        other = np.random.choice(matingPool)
                        while other != better:
                            other = np.random.choice(matingPool)
                        for b in self.imgs[("photo_"+str(other))]["DNA"][crossPoint:]:
                            ww.append(b)
                    else:
                        other = np.random.choice(matingPool)
                        while other != better:
                            other = np.random.choice(matingPool)
                        for f in self.imgs[("photo_"+str(other))]["DNA"][:crossPoint]:
                            ww.append(f)
                        for b in self.imgs[("photo_"+str(better))]["DNA"][crossPoint:]:
                            ww.append(b)

                    for w2 in range(len(ww)):
                        if np.random.rand() >= 0.1:
                            max1 = ww[w].max()
                            indexMax = np.where(ww[w2] == max1)

                            if max1 >= 0:
                                ww[w2][indexMax] = max1 * np.random.rand()
                            else:
                                ww[w2][indexMax] = max1 * -np.random.rand()

                if r == 0:
                    self.imgs[("photo_"+str(r*5+c))]["DNA"] = ww
                    self.cycle.generator_photo.set_weights(self.imgs[("photo_"+str(r*5+c))]["DNA"])
                    self.photo = self.cycle.train(draw=self.npImg)
                    self.imgs[("photo_"+str(r*5+c))]["photo"] = FigureCanvasTkAgg(self.photo, self.photos)
                    self.imgs[("photo_"+str(r*5+c))]["points"]["text"] = 0

                    self.imgs[("photo_"+str(r*5+c))]["photo"].get_tk_widget().grid(row=r, column=c)

                if r == 2:
                    self.imgs[("photo_"+str((r-1)*5+c))]["DNA"] = ww
                    self.cycle.generator_photo.set_weights(self.imgs[("photo_"+str((r-1)*5+c))]["DNA"])
                    self.photo = self.cycle.train(draw=self.npImg)
                    self.imgs[("photo_"+str((r-1)*5+c))]["photo"] = FigureCanvasTkAgg(self.photo, self.photos)
                    self.imgs[("photo_"+str((r-1)*5+c))]["points"]["text"] = 0

                    self.imgs[("photo_"+str((r-1)*5+c))]["photo"].get_tk_widget().grid(row=r, column=c)

                root.update()

        self.photos.pack(side=tk.LEFT)

    def widgetsNextStep(self):
        self.photos = tk.Frame(self.master, padx=10, pady=20)

        self.imgs = {}

        self.butNextGen = ttk.Button(self.photos)
        self.butNextGen["text"] = "Carregar Nova Geração"
        self.butNextGen["width"] = 25
        self.butNextGen["command"] = self.NextGen
        self.butNextGen.grid(row=4, column=2)

        self.cycle.ckpt.restore(
            self.cycle.ckpt_manager.latest_checkpoint).expect_partial()

        for r in range(4):
            for c in range(5):
                tmpDict = { "photo": None,
                        "DNA": np.array(self.cycle.generator_photo.get_weights()),
                        "points": 0}

                ww = []
                for w in tmpDict["DNA"]:
                    max1 = w.max()
                    indexMax = np.where(w == max1)

                    if max1 >= 0:
                        w[indexMax] = max1 * np.random.rand()
                    else:
                        w[indexMax] = max1 * -np.random.rand()

                    ww.append(w)

                tmpDict["DNA"] = ww
                self.cycle.generator_photo.set_weights(tmpDict["DNA"])
                self.photo = self.cycle.train(draw=self.npImg)
                tmpDict["photo"] = FigureCanvasTkAgg(self.photo, self.photos)

                if r == 0:
                    self.imgs[("photo_"+str(r*5+c))] = tmpDict
                    self.imgs[("photo_"+str(r*5+c))]["points"] = tk.Label(self.photos, text="0", font=("arial 10"))

                    self.imgs[("photo_"+str(r*5+c))]["photo"].get_tk_widget().grid(row=r, column=c)
                if r == 1:
                    self.imgs[("photo_"+str((r-1)*5+c))]["points"].grid(row=r, column=c)
                if r == 2:
                    self.imgs[("photo_"+str((r-1)*5+c))] = tmpDict
                    self.imgs[("photo_"+str((r-1)*5+c))]["points"] = tk.Label(self.photos, text="0", font=("arial 10"))

                    self.imgs[("photo_"+str((r-1)*5+c))]["photo"].get_tk_widget().grid(row=r, column=c)
                if r == 3:
                    self.imgs[("photo_"+str((r-2)*5+c))]["points"].grid(row=r, column=c)

                root.update()

        self.photos.pack(side=tk.LEFT)

        while not exitFlag:
            try:
                self.mx, self.my = root.winfo_pointerxy()

                for i in range(10):
                    self.imgx = self.imgs[("photo_"+str(i))]["photo"].get_tk_widget().winfo_rootx()
                    self.imgy = self.imgs[("photo_"+str(i))]["photo"].get_tk_widget().winfo_rooty()

                    if (0 < (self.mx - self.imgx) < 256) and (0 < (self.my - self.imgy) < 256):
                        self.imgs[("photo_"+str(i))]["points"]["text"] = int(self.imgs[("photo_"+str(i))]["points"]["text"]) + 1

                root.update()

            except Exception as e:
                print(e)

def on_quit():
    global exitFlag
    exitFlag = True
    root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    main(root)
    root.title("Teste")
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()
