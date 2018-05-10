from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
import time
import threading

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import resources as res

"""
WRITTEN BY
Micheal Kirkegaard
"""

class AppUI(Tk):
	def __init__(self, w=1280, h=720):
		Tk.__init__(self)
		self.geometry("{0}x{1}".format(w,h))

		# Var
		self.plot_data = None
		self.progbarThread = None

		# Slave Widgets 
		self.main_frame = self.Main_Frame(self, self, w, h)

		# Grid
		self.main_frame.grid(sticky=N+E+S+W)

		# Menu
		self.menubar = self.__topmenubar()
		self.config(menu=self.menubar)

		# Shortcuts
		self.bind('<Shift-N>', lambda e: self.New_File())
		self.bind('<Shift-O>', lambda e: self.Open_File())
		self.bind('<Shift-S>', lambda e: self.Save_File())
		self.bind('<Shift-C>', lambda e: self.Close_File())
		self.bind('<Shift-Escape>', lambda e: self.quit())

		# Init
		self.mainloop()
	
	def __topmenubar(self):
		menubar = Menu(self)

		# Filemenu
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="New", command=self.New_File, accelerator='Shift-N')
		filemenu.add_command(label="Open", command=self.Open_File, accelerator='Shift-O')
		filemenu.add_command(label="Save", command=self.Save_File, accelerator='Shift-S')
		filemenu.add_command(label="Close", command=self.Close_File, accelerator='Shift-Enter')
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit, accelerator='Shift-Escape')
		menubar.add_cascade(label="File", menu=filemenu)

		# Helpmenu
		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About", command=lambda: self.popup(res.hm_ABOUT_TEXT))
		helpmenu.add_command(label="File Formats", command=lambda: self.popup(res.hm_FORMAT_TEXT))
		helpmenu.add_command(label="Commands", command=lambda: self.popup(res.hm_COMMANDS_TEXT))
		menubar.add_cascade(label="Help", menu=helpmenu)

		return menubar

	# New File
	def New_File(self):
		if not self.progbarThread:
			def task(toplevel, filepath_edf, filepath_anno, pb, b):
				dest = False
				try:
					# Mockup file
					size = 1000
					for _ in range(size):
						# Soft Close
						if self.progbarThread.getName() in ['cancel','close']: # Shutdown Flags
							raise()
						# Do files and stuff
						time.sleep(1.0/size) 
						# step out of 100%
						pb.step(100/size)

					self.plot_Data = None
					self.main_frame.open_plot(self.plot_data)
					dest = True
				except Exception as e:
					if self.progbarThread.getName() != 'close':
						self.Close_File()
				finally:
					b.grid_forget()
					pb.grid_forget()
					self.progbarThread = None
					self.unbind('<Escape>')
					if dest:
						toplevel.destroy()

			def canceltask(b_go):
				if self.progbarThread and self.progbarThread.is_alive() and self.progbarThread.getName() != 'cancel':
					self.progbarThread.setName('cancel') # Raise Flag
					b_go.grid()

			def getFilePath(svar, filetitle, filetag, callback, callbackarg):
				try:
					filepath = filedialog.askopenfilename(title='Choose '+filetitle+' File', filetypes=[(filetitle,'*'+filetag)])
					if not filepath or filepath == '':
						raise()
					svar.set(filepath)
				except Exception as e:
					svar.set("Error Loading File...")
				finally:
					callback(callbackarg)
			
			def starttask(toplevel, b_go, filepath_edf, filepath_anno):
				if not self.progbarThread:
					#TODO: CHECK isfile(path)

					self.Close_File()
					b_go.grid_forget()

					# Progbar
					pb = Progressbar(toplevel)
					pb.grid(row=3, column=0)
					self.bind('<Escape>', lambda e: canceltask(b_go))

					# Cancel Button
					b = Button(toplevel, text='Cancel')
					orig_color = b.cget("background")
					b.bind('<Button-1>', lambda event: canceltask(b_go))
					b.grid(row=3, column=1)

					# Task
					self.progbarThread = threading.Thread(target=task, args=(toplevel, filepath_edf, filepath_anno, pb, b))
					self.progbarThread.setName('start')
					self.progbarThread.start()

			def focus(toplevel):
				toplevel.lift()
				toplevel.focus_force()
				toplevel.grab_set()
				toplevel.grab_release()

			file_toplevel = Toplevel()
			focus(file_toplevel)
			# EDF file
			filepath_edf = StringVar(value='Choose File...')
			entry_edf = Entry(file_toplevel, textvariable=filepath_edf, width=80)
			b_edf = Button(file_toplevel, text='Choose File')
			b_edf.bind('<Button-1>', lambda e: getFilePath(filepath_edf, res.ff_FILETITLE_e, res.ff_FILETAG_e, focus, file_toplevel))

			# ANN file
			filepath_anno = StringVar(value='Choose File...')
			entry_anno = Entry(file_toplevel, textvariable=filepath_anno, width=80)
			b_anno = Button(file_toplevel, text='Choose File')
			b_anno.bind('<Button-1>', lambda e: getFilePath(filepath_anno, res.ff_FILETITLE_a, res.ff_FILETAG_a, focus, file_toplevel))

			# Go button
			b_go = Button(file_toplevel, text='Go')

			# Grids
			entry_edf.grid(row=0, column=0)
			b_edf.grid(row=0, column=1)

			entry_anno.grid(row=1, column=0)
			b_anno.grid(row=1, column=1)

			b_go.grid(sticky=W+E)
			b_go.bind('<Button-1>', lambda e: starttask(file_toplevel, b_go, filepath_edf.get(),filepath_anno.get()))

			# Binds
			file_toplevel.bind('<Escape>', lambda e: file_toplevel.destroy())
			file_toplevel.bind('<Return>', lambda e: starttask(file_toplevel, b_go, filepath_edf.get(),filepath_anno.get()))			

	# Open already formatted plots
	def Open_File(self):
		if not self.progbarThread:
			try:
				# TODO: Default dir
				file = filedialog.askopenfile(title='Choose '+ res.ff_FILETITLE_a +' file', filetypes=[(res.ff_FILETITLE_a,'*'+res._FILETAG_a)])
				self.plot_Data = None # TODO: pickle.dump(file)
				file.close()
			except Exception as e:
				return
			self.main_frame.open_plot(self.plot_data)
	
	# Save plotfile
	def Save_File(self):
		if self.plot_Data:
			# TODO: Default dir
			file = filedialog.asksaveasfile(filetypes=[(res.ff_FILETITLE_a,'*'+res.ff_FILETAG_a)])
			# TODO: pickle.dump(file, self.plot_Data)
			file.close()

	# Close/Cancel
	def Close_File(self):
		if self.progbarThread and self.progbarThread.is_alive() and self.progbarThread.getName() != 'close':
			self.progbarThread.setName('close') # Raise Flag
		self.main_frame.close_plot()
		self.plot_Data = None

	def popup(self, text):
		toplevel = Toplevel()
		# Title
		for s in text[:2]:
			label = Label(toplevel, text=s, font=res.FONT)
			label.grid(sticky=N)

		# Labels
		gridsize = [max([len(t) for t in tub]) for tub in zip(*[s for s in text if isinstance(s, list)])][:-1]
		for i,s in enumerate(text[2:]):
			if isinstance(s, list):
				subframe = Frame(toplevel)
				for j,ss in enumerate(s):
					label = Label(subframe, text=ss, width=gridsize[j] if j < len(gridsize) else None, font=res.FONT, anchor=W)
					label.grid(row=0, column=j)
				subframe.grid(sticky=N+W)
			else:
				label = Label(toplevel, text=s, font=res.FONT, anchor=W)
				label.grid(sticky=N+W)

		# Focus grab
		toplevel.bind('<Escape>', lambda e: toplevel.destroy())
		toplevel.bind('<Return>', lambda e: toplevel.destroy())
		toplevel.bind('<Shift-C>', lambda e: toplevel.destroy())
		toplevel.lift()
		toplevel.focus_force()
		toplevel.grab_set()
		toplevel.grab_release()

	class Main_Frame(Frame):
		def __init__(self, master, controller, w=None, h=None):
			Frame.__init__(self, master, width=w, height=h) # Super.__init__()
			self.controller = controller
			self.plot_data = None

			# Slaves
			self.plot_frame = self.Plot_Frame(self, controller)
			self.prop_frame = self.Prop_Frame(self, controller)

			# Grid
			self.plot_frame.grid(row=0, column=0)
			self.prop_frame.grid(row=1, column=0)

			# Default 
			self.close_plot()

		# Show plotfile
		def open_plot(self, plot_data=None):
			#if plot_data:
				self.plot_data = plot_data
				self.plot_frame.grid()
				self.prop_frame.grid()
				# TODO: Plots n' stuff
				# TODO: Properties n' stuff

		# Close plotfile
		def close_plot(self):
			self.plot_frame.grid_forget()
			self.prop_frame.grid_forget()

		class Plot_Frame(Frame):
			def __init__(self, master, controller):
				Frame.__init__(self, master) # Super.__init__()
				self.controller = controller

				# Widget Packing
				self.__plot_menubar().grid(row=0, column=0)
				self.__plot_main().grid(row=1, column=0)

			# plot menus
			def __plot_menubar(self):
				subframe = Frame(self)
				b_0 = Button(subframe, text='Raw ECG')
				b_1 = Button(subframe, text='')
				b_2 = Button(subframe, Text='')
				
				b_0.bind('<Button-1>', None) # Filter plot
				return subframe

			# main plt
			def __plot_main(self):
				subframe = Frame(self)

				# https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/

				w=1280
				h=720
				dpi = 100
				f = Figure(figsize=(w/dpi, h/dpi), dpi=dpi)
				a = f.add_subplot(111)
				a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

				canvas = FigureCanvasTkAgg(f, subframe)
				canvas.show()
				canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

				toolbar = NavigationToolbar2TkAgg(canvas, subframe)
				toolbar.update()
				canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

				return subframe

		class Prop_Frame(Frame):
			def __init__(self, master, controller):
				Frame.__init__(self, master) # Super.__init__()
				self.controller = controller

				# Widget slave packing
				self.__plot_properties().grid()
	
			# Properties
			def __plot_properties(self):
				subframe = Frame(self)
				Label(subframe, text='plot_properties').pack()
				# Raw
				# Raw featues
				# Final features
				# arousals
				return subframe

if __name__ == '__main__':
	App = AppUI()