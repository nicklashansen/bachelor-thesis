from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar, Separator
import time
import threading
#from dataflow import dataflow
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import resources as res

"""
WRITTEN BY
Micheal Kirkegaard
"""

w_=1280-180
h_=720
dpi_ = 100
f_ = Figure(figsize=(w_/dpi_, h_/dpi_), dpi=dpi_)
a_ = f_.add_subplot(111)
a_.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

TEST_FIG = f_
TEST_PROP = {0:0, 1:1, 2:2, 3:3, 4:4}
TEST_PROP_2 = {5:5, 6:6, 7:7, 8:8, 9:9}

class AppUI(Tk):
	def __init__(self, w=1280, h=720):
		Tk.__init__(self)
		# Setup
		self.geometry("{0}x{1}".format(w,h))
		self.resizable(False, False)
		self.width = w
		self.height = h

		# Plot Vars
		self.plot_data = []
		self.btn_plot_states = []

		# Prop vars
		self.property_dict = {}

		# New File Thread
		self.progbarThread = None

		# Slave Widgets 
		self.main_frame = self.Main_Frame(self, self)

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
			def task(toplevel, edf, anno, pb, b):
				destroy = False
				try:
					# Mockup file
					size = 1000
					for _ in range(size):
						# Soft Close
						if self.progbarThread.getName() in ['cancel','close']: # Shutdown Flags
							raise
						# Do files and stuff
						time.sleep(1.0/size) 
						# step out of 100%
						pb.step(100/size)#'''
					
					# ---
					# 1) Get Signals (plot is renderes elsewhere)
					# 2) find propeties from signals
					# 3) -->
					# ---

					plot_data = None
					property_dict = TEST_PROP_2

					# Finished
					destroy = True
				except Exception as e:
					pass
				finally:
					self.progbarThread = None
					self.unbind('<Escape>')
					if destroy:
						toplevel.destroy()
						self.Close_File()
						self.Show_Plot(plot_data, property_dict)
					else:
						b.grid_forget()
						pb.grid_forget()

			def canceltask(toplevel,b_go):
				if self.progbarThread and self.progbarThread.is_alive() and self.progbarThread.getName() != 'cancel':
					self.progbarThread.setName('cancel') # Raise Flag
					b_go.grid()
					toplevel.bind('<Escape>', lambda e: toplevel.destroy())
			
			def starttask(toplevel, b_go, filepath_edf, filepath_anno):
				if not self.progbarThread:
					#TODO: CHECK path and filetype

					b_go.grid_remove()
					toplevel.unbind('<Escpae>')
					toplevel.bind('<Escape>', lambda e: canceltask(toplevel, b_go))

					# Progbar
					pb = Progressbar(toplevel)
					pb.grid(row=3, column=0)
					pb.columnconfigure(0, weight=2)

					# Cancel Button
					b = Button(toplevel, text='Cancel')
					orig_color = b.cget("background")
					b.bind('<Button-1>', lambda event: canceltask(toplevel, b_go))
					b.grid(row=3, column=2, sticky=NSEW)

					# Task
					self.progbarThread = threading.Thread(target=task, args=(toplevel, filepath_edf, filepath_anno, pb, b))
					self.progbarThread.setName('start')
					self.progbarThread.start()
			
			def getFilePath(svar, filetitle, filetag, callback):
				try:
					filepath = filedialog.askopenfilename(title='Choose '+filetitle+' File', filetypes=[(filetitle,'*'+filetag)])
					if not filepath or filepath == '':
						raise()
					svar.set(filepath)
				except Exception as e:
					# TODO: ErrorMsg
					svar.set("Error Loading File...")
				finally:
					callback()

			def focus(toplevel):
				toplevel.lift()
				toplevel.focus_force()
				toplevel.grab_set()
				#toplevel.grab_release()

			file_toplevel = Toplevel()
			focus(file_toplevel)

			# EDF file
			filepath_edf = StringVar(value='Choose File...')
			label_edf = Label(file_toplevel, text=res.ff_FILETITLE_e+':', anchor=E)
			entry_edf = Entry(file_toplevel, textvariable=filepath_edf, width=80)
			b_edf = Button(file_toplevel, text='Choose File', command=lambda: getFilePath(filepath_edf, res.ff_FILETITLE_e, res.ff_FILETAG_e, lambda: focus(file_toplevel)))

			# ANN file
			filepath_anno = StringVar(value='Choose File...')
			label_anno = Label(file_toplevel, text=res.ff_FILETITLE_s+':', anchor=E)
			entry_anno = Entry(file_toplevel, textvariable=filepath_anno, width=80)
			b_anno = Button(file_toplevel, text='Choose File', command=lambda: getFilePath(filepath_anno, res.ff_FILETITLE_s, res.ff_FILETAG_s, lambda: focus(file_toplevel)))

			# Go button
			b_go = Button(file_toplevel, text='Go')
			b_go.bind('<Button-1>', lambda e: starttask(file_toplevel, b_go, filepath_edf.get(), filepath_anno.get()))	

			# Grids
			label_edf.grid(row=0, column=0, sticky=E)
			entry_edf.grid(row=0, column=1)
			b_edf.grid(row=0, column=2)
			
			label_anno.grid(row=1, column=0, sticky=E)
			entry_anno.grid(row=1, column=1)
			b_anno.grid(row=1, column=2)

			b_go.grid(row=2, column=2, sticky=NSEW)

			# Binds
			file_toplevel.bind('<Return>', lambda e: starttask(file_toplevel, b_go, filepath_edf.get(), filepath_anno.get()))
			file_toplevel.bind('<Escape>', lambda e: file_toplevel.destroy())			

	# Open already formatted plots
	def Open_File(self):
		if not self.progbarThread:
			try:
				# TODO: Default dir
				file = filedialog.askopenfile(title='Choose '+ res.ff_FILETITLE_a +' file', filetypes=[(res.ff_FILETITLE_a,'*'+res.ff_FILETAG_a)])

				# ---
				# # x = [self.plot_data, self.property_dict]
				# x = "pickle read(file)"
				# plot_data = x[0]
				# property_dict = x[1]
				# ---

				plot_data = None
				property_dict = TEST_PROP	

				file.close()
			except Exception as e:
				# TODO: ErrorMsg
				return

			self.Close_File()
			self.Show_Plot(plot_data, property_dict)
	
	def Show_Plot(self, plot_data, property_dict):
		self.main_frame.close_plot()
		self.plot_data = plot_data
		self.property_dict = property_dict
		self.main_frame.open_plot()

	# Save plotfile
	def Save_File(self):
		if self.plot_data:
			try:
				# TODO: Default dir
				file = filedialog.asksaveasfile(filetypes=[(res.ff_FILETITLE_a,'*'+res.ff_FILETAG_a)])
				
				# ---
				# x = [self.plot_data, self.property_dict]
				# pickle.dump(file, x)
				# ---

				file.close()
			except Exception as e:
				# TODO: ErrorMsg
				return 

	# Close
	def Close_File(self):
		self.main_frame.close_plot()
		self.plot_data = []
		self.property_dict = {}
	
	# Text Popup
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
	
	# press button
	def plot_btn_state_swap(self, btn, id):
		# button state swap
		if (btn['text'][-2:] == 'ON'):
			btn['text'] = btn['text'][:-2]+'OFF'
		else:
			btn['text'] = btn['text'][:-3]+'ON'

		self.btn_plot_states[id] = not self.btn_plot_states[id] 
		self.main_frame.update_plot()

	class Main_Frame(Frame):
		def __init__(self, master, controller):
			Frame.__init__(self, master) # Super.__init__()
			self.controller = controller
			
			# Slaves
			self.plot_frame = self.Plot_Frame(self, controller)
			self.prop_frame = self.Prop_Frame(self, controller)

			# Grid
			self.plot_frame.grid(row=0, column=0)
			self.prop_frame.grid(row=0, column=1, sticky=NE)

			# Default
			self.close_plot()

		# Show plot
		def open_plot(self):
			self.plot_frame.update_menu()
			self.plot_frame.update_plot()
			self.plot_frame.grid()
			
			self.prop_frame.update_properties()
			self.prop_frame.grid()

		# Update plot
		def update_plot(self):
			self.plot_frame.update_plot()

		# Close plot
		def close_plot(self):
			self.plot_frame.grid_remove()
			self.prop_frame.grid_remove()

		class Plot_Frame(Frame):
			def __init__(self, master, controller):
				Frame.__init__(self, master,bg='white') # Super.__init__()
				self.controller = controller

				# Widget Packing
				self.menu = None
				self.plot = None

				# Grid
				self.update_menu()
				self.update_plot()

			# plot menus
			def update_menu(self):
				subframe = Frame(self)

				w = 25

				# Buttons
				b_0 = Button(subframe, width=w, text='Raw ECG : ON')
				self.controller.btn_plot_states += [True]
				b_0['command'] = lambda: self.controller.plot_btn_state_swap(b_0, 0)

				b_1 = Button(subframe, width=w, text='Raw PPG : OFF')
				self.controller.btn_plot_states += [False]
				b_1['command'] = lambda:self.controller.plot_btn_state_swap(b_1, 1)

				b_2 = Button(subframe, width=w, text='Arousals : ON')
				self.controller.btn_plot_states += [True]
				b_2['command'] = lambda: self.controller.plot_btn_state_swap(b_2, 2)

				b_3 = Button(subframe, width=w, text='Wake State Regions: ON')
				self.controller.btn_plot_states += [True]
				b_3['command'] = lambda: self.controller.plot_btn_state_swap(b_3, 3)

				b_4 = Button(subframe, width=w, text='REM state Regions: ON')
				self.controller.btn_plot_states += [True]
				b_4['command'] = lambda: self.controller.plot_btn_state_swap(b_4, 4)

				b_5 = Button(subframe, width=w, text='nREM state Regions: ON')
				self.controller.btn_plot_states += [True]
				b_5['command'] = lambda: self.controller.plot_btn_state_swap(b_5, 5)

				# Grid
				b_0.grid(row=0, column=0, sticky=W)
				b_1.grid(row=0, column=1, sticky=W)
				b_2.grid(row=0, column=2, sticky=W)
				b_3.grid(row=0, column=3, sticky=W)
				b_4.grid(row=0, column=4, sticky=W)
				b_5.grid(row=0, column=5, sticky=W)

				if(self.menu):
					self.menu.grid_forget()
				self.menu = subframe
				subframe.grid(row=0, column=0, sticky=N)

			# update plot
			def update_plot(self):
				subframe = Frame(self)

				f = TEST_FIG if not self.controller.plot_data else None #TODO: plot_compute(self.controller.plot_data)
				
				#w_=self.controller.width - 180
				#h_=self.controller.height - 60
				#dpi_ = 100
				#f = Figure(figsize=(w_/dpi_, h_/dpi_), dpi=dpi_)
				#a = f.add_subplot(111)
				#a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

				canvas = FigureCanvasTkAgg(f, subframe)
				canvas.show()
				canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

				toolbar = NavigationToolbar2TkAgg(canvas, subframe)
				toolbar.update()
				toolbar.config(background='white')
				canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

				if(self.plot):
					self.plot.grid_forget()
				self.plot = subframe
				self.plot.grid(row=1, column=0)

		class Prop_Frame(Frame):
			def __init__(self, master, controller):
				Frame.__init__(self, master, bg='white') # Super.__init__()
				self.controller = controller
				self.properties = []
				self.update_properties()
			
			# Update
			def update_properties(self):
				if self.properties != None:
					for frame in self.properties:
						frame.grid_forget()

				for k,v in self.controller.property_dict.items():
					subframe = self.__property(str(k), str(v), 9, res.FONT)
					subframe.grid(sticky=NE)
					self.properties += [subframe]

			# Properties
			def __property(self, key, val, width, font):
				subframe = Frame(self)
				Label(subframe, text=key, font=font, width=width).grid(row=0, column=0)
				Separator(subframe, orient=VERTICAL).grid(row=0, column=1, rowspan=3, sticky=NS)
				Label(subframe, text=val, font=font, width=width).grid(row=0, column=2)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=0, rowspan=3, sticky=EW)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=1, rowspan=3, sticky=EW)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=2, rowspan=3, sticky=EW)
				return subframe

if __name__ == '__main__':
	App = AppUI()