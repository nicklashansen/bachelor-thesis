'''
WRITTEN BY:
Michael Kirkegaard

MAIN PURPOSE:
This class contains the code for the prototype of a visual tool, which can aid in diagnosing sleep appnea by visualising features
of a Gated Recurrent Unit model and its predicted arousals as a set of 6 subplots as well as showing statistic properties of them.
'''
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar, Separator
from tkinter import messagebox
import time
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import resources as res
import sys, os
if __name__ == '__main__':
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots import plot_results
from preprocessing import prep_X
import filesystem as fs
import traceback


class AppUI(Tk):
	'''
	Main controller class of the GUI. It controls and maps all frame resources
	and pop-ups together by controlling variables and navigation between windows.
	'''
	
	# Initilisation, will attempt to load last plot displayed before last application shutodwn.
	def __init__(self, w=res.ss_WIDTH, h=res.ss_HEIGHT):
		Tk.__init__(self)	
		# Setup
		self.geometry("{0}x{1}".format(w,h))
		self.resizable(False, False)
		self.width = w
		self.height = h

		# Plot Vars
		self.plot_data = []
		self.plot_figure = None
		self.btn_plot_states = []
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
		self.bind('<'+res.fm_sc_NEW+'>', lambda e: self.New_File())
		self.bind('<'+res.fm_sc_LOAD+'>', lambda e: self.Load_File())
		self.bind('<'+res.fm_sc_SAVE+'>', lambda e: self.Save_File())
		self.bind('<'+res.fm_sc_CLOSE+'>', lambda e: self.Close_File())
		self.bind('<'+res.fm_sc_EXIT+'>', lambda e: self.quit())

		# Load last plot
		try:
			if os.path.isfile(fs.Filepaths.TempAplotFile):
				plot_data, property_dict = fs.load_aplot(fs.Filepaths.TempAplotFile)
				property_dict = [('aplot_path', fs.Filepaths.TempAplotFile.replace('\\','/'))] + property_dict
				self.Show_Plot(plot_data, property_dict)
		except Exception as e:
			self.Close_File()
			ex = traceback.format_exc()
			messagebox.showwarning("Warning", "Error loading latest aplot file.\n\n"+str(e)+'\n'+ex)
	
	# Top menubar mappings
	def __topmenubar(self):
		menubar = Menu(self)

		# Filemenu
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="New", command=self.New_File, accelerator=res.fm_sc_NEW)
		filemenu.add_command(label="Load", command=self.Load_File, accelerator=res.fm_sc_LOAD)
		filemenu.add_command(label="Save", command=self.Save_File, accelerator=res.fm_sc_SAVE)
		filemenu.add_command(label="Close", command=self.Close_File, accelerator=res.fm_sc_CLOSE)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit, accelerator=res.fm_sc_EXIT)
		menubar.add_cascade(label="File", menu=filemenu)

		# Helpmenu
		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About", command=lambda: self.popup(res.hm_ABOUT_TEXT))
		helpmenu.add_command(label="File Formats", command=lambda: self.popup(res.hm_FORMAT_TEXT))
		helpmenu.add_command(label="Commands", command=lambda: self.popup(res.hm_COMMANDS_TEXT))
		menubar.add_cascade(label="Help", menu=helpmenu)

		return menubar

	def New_File(self):
		'''
		Controls performing a new analysis on a PSG recording and annotation file. It creates a toplevel window from which the
		user can choose the two files used for the analysis. The anlysis runs in its own thread and can be canceled at any time.
		File formats will be verified and forced by file-picker window. An Error is displayed if the analysis fails at any step.
		'''

		# Cannot run two new files operations at the same time
		if not self.progbarThread:

			# Perform the analysis task
			def task(toplevel, edf_e, edf_b, anno_e, anno_b, b_go, edf, anno, cancelbutton, statuslabel, progresbar):
				this = self.progbarThread
				destroy = False
				try:
					steps = 5
					# Step 0 Load files
					statuslabel['text'] = 'Loading files...'
					if this.getName() == res.SHUTDOWNFLAG: # Shutdown Flags
						raise
					if not fs.validate_fileformat(edf, anno):
						raise Exception('File corrupt or incorrect format.')

					# Step 1 prep files
					progresbar.step(int(100/steps))
					statuslabel['text'] = 'Preprocessing files...'
					if this.getName() == res.SHUTDOWNFLAG: # Shutdown Flags
						raise
					X = prep_X(edf, anno)
					#X,_ = fs.load_csv(edf[-19:-4])

					# Step 2 Tensorflow
					progresbar.step(int(100/steps))
					statuslabel['text'] = 'Loading keras and tensorflow...'
					if this.getName() == res.SHUTDOWNFLAG: # Shutdown Flags
						raise
					from dataflow import dataflow

					# Step 3 - Analyse
					progresbar.step(int(100/steps))
					statuslabel['text'] = 'Analysing data...'
					if this.getName() in res.SHUTDOWNFLAG: # Shutdown Flags
						raise
					plot_data, property_dict = dataflow(X)
					property_dict = [('edf_path',edf),('anno_path',anno)] + property_dict

					# Step 4 - Plot data
					progresbar.step(int(100/steps))
					statuslabel['text'] = 'Plotting results...'
					if this.getName() == res.SHUTDOWNFLAG: # Shutdown Flags
						raise

					# step 5 - Finished
					self.progbarThread = None
					self.unbind('<Escape>')
					self.Close_File()
					self.Show_Plot(plot_data, property_dict)
					toplevel.destroy()
				
				except Exception as e:
					if(this.getName() != res.SHUTDOWNFLAG):
						ex = traceback.format_exc()
						messagebox.showerror("Error", res.ERROR_MSG(e, ex, statuslabel['text']))
				finally:
					if(this.getName() != res.SHUTDOWNFLAG):
						canceltask(toplevel, edf_e, edf_b, anno_e, anno_b, b_go,cancelbutton,statuslabel,progresbar)
			
			# Cancel the analysis by raising shutdown flag for task-thread, releasing input entries
			def canceltask(toplevel, edf_e, edf_b, anno_e, anno_b, b_go,cancelbutton,statuslabel,progresbar):
				if self.progbarThread:
					# Forget task
					self.progbarThread.setName(res.SHUTDOWNFLAG)
					self.progbarThread = None

					# Reset view
					edf_e['state'] = 'normal'
					edf_b['state'] = 'normal'
					anno_e['state'] = 'normal'
					anno_b['state'] = 'normal'
					b_go.grid()

					self.unbind('<Escape>')
					toplevel.bind('<Escape>', lambda e: toplevel.destroy())

					cancelbutton.grid_forget()
					statuslabel.grid_forget()
					progresbar.grid_forget()
			
			# Begin the analysis, locking input entries
			def starttask(toplevel, edf_e, edf_b, anno_e, anno_b, b_go, filepath_edf, filepath_anno):
				if not self.progbarThread:

					edf_e['state'] = 'disabled'
					edf_b['state'] = 'disabled'
					anno_e['state'] = 'disabled'
					anno_b['state'] = 'disabled'

					# Progbar
					pb = Progressbar(toplevel)
					pb.grid(row=3, column=0)

					# StatusLabel
					sl = Label(toplevel)
					sl.grid(row=3, column=1)

					# Cancel Button
					b_go.grid_remove()
					b = Button(toplevel, text='Cancel')
					orig_color = b.cget("background")
					b.bind('<Button-1>', lambda event: canceltask(toplevel, edf_e, edf_b, anno_e, anno_b, b_go, b, sl, pb))
					b.grid(row=3, column=2, sticky=NSEW)

					# Bind
					toplevel.bind('<Escape>', lambda e: canceltask(toplevel, edf_e, edf_b, anno_e, anno_b, b_go, b, sl, pb))

					# Task
					self.progbarThread = threading.Thread(target=task, args=(toplevel, edf_e, edf_b, anno_e, anno_b, b_go, filepath_edf, filepath_anno, b, sl, pb))
					self.progbarThread.start()
			
			# Opens file-picker window, enforcing format of file
			def getFilePath(svar, filetitle, filetag, callback):
				try:
					filepath = filedialog.askopenfilename(title='Choose '+filetitle+' File', filetypes=[(filetitle,'*'+filetag)])
					if not filepath or filepath == '':
						return
					svar.set(filepath)
				except Exception as e:
					ex = traceback.format_exc()
					messagebox.showerror("Error", res.ERROR_MSG(e, ex))
					svar.set("Error Loading File...")
				finally:
					callback()
			
			# Forced focus of new file wub window
			def focus(toplevel):
				toplevel.lift()
				toplevel.focus_force()
				toplevel.grab_set()

			# Creates sub-window
			file_toplevel = Toplevel()
			focus(file_toplevel)

			# EDF file entry
			filepath_edf = StringVar(value='Choose File...')
			label_edf = Label(file_toplevel, text=res.ff_FILETITLE_e+':', anchor=E)
			entry_edf = Entry(file_toplevel, textvariable=filepath_edf, width=res.ss_ENTRY_WIDTH)
			b_edf = Button(file_toplevel, text='Choose File', command=lambda: getFilePath(filepath_edf, res.ff_FILETITLE_e, res.ff_FILETAG_e, lambda: focus(file_toplevel)))

			# ANN file entry
			filepath_anno = StringVar(value='Choose File...')
			label_anno = Label(file_toplevel, text=res.ff_FILETITLE_s+':', anchor=E)
			entry_anno = Entry(file_toplevel, textvariable=filepath_anno, width=res.ss_ENTRY_WIDTH)
			b_anno = Button(file_toplevel, text='Choose File', command=lambda: getFilePath(filepath_anno, res.ff_FILETITLE_s, res.ff_FILETAG_s, lambda: focus(file_toplevel)))

			# Go button
			b_go = Button(file_toplevel, text='Go')
			b_go.bind('<Button-1>', lambda e: starttask(file_toplevel, entry_edf, b_edf, entry_anno, b_anno, b_go, filepath_edf.get(), filepath_anno.get()))	

			# Grids
			label_edf.grid(row=0, column=0, sticky=E)
			entry_edf.grid(row=0, column=1)
			b_edf.grid(row=0, column=2)
			
			label_anno.grid(row=1, column=0, sticky=E)
			entry_anno.grid(row=1, column=1)
			b_anno.grid(row=1, column=2)

			b_go.grid(row=2, column=2, sticky=NSEW)

			# Binds
			file_toplevel.bind('<Return>', lambda e: starttask(file_toplevel, entry_edf, b_edf, entry_anno, b_anno, b_go, filepath_edf.get(), filepath_anno.get()))
			file_toplevel.bind('<Escape>', lambda e: file_toplevel.destroy())
			
			# Top level position
			x = self.winfo_x()
			y = self.winfo_y()
			file_toplevel.geometry("+%d+%d" % (x+5, y+55))		

	def Load_File(self):
		'''
		Loads an already analysed plot file of correct format enforced by tKinter file-picker. An Error is displayed if this fails.
		'''
		if not self.progbarThread:
			try:
				filepath = filedialog.askopenfilename(title='Choose '+ res.ff_FILETITLE_a +' file', filetypes=[(res.ff_FILETITLE_a,'*'+res.ff_FILETAG_a)])
				if not filepath or filepath == '':
					return
				plot_data, property_dict = fs.load_aplot(filepath)
				property_dict = [('aplot_path',filepath)] + [x for x in self.property_dict if x[0] != 'aplot_path']
			except Exception as e:
				ex = traceback.format_exc()
				messagebox.showerror("Error", res.ERROR_MSG(e, ex))
				return
			self.Close_File()
			self.Show_Plot(plot_data, property_dict)

	# Save plotfile
	def Save_File(self):
		'''
		Save the currently analysed plot file of file is format enforced by tKinter file-picker.
		An Error is displayed if this fails, or an message about succesful saving if it does not.
		'''
		if self.plot_data and self.property_dict is not None:
			try:
				filepath = filedialog.asksaveasfilename(filetypes=[(res.ff_FILETITLE_a,'*'+res.ff_FILETAG_a)])
				if not filepath or filepath == '':
					return
				save_property_dict = [x for x in self.property_dict if x[0] not in  ['aplot_path', 'edf_path', 'anno_path']]
				fs.write_aplot(filepath, self.plot_data, save_property_dict)
				self.property_dict = [('aplot_path',filepath)] + [x for x in self.property_dict if x[0] != 'aplot_path']
				self.main_frame.prop_frame.update_properties()
				messagebox.showinfo("Succes", "Succesfully saved file.")
			except Exception as e:
				ex = traceback.format_exc()
				messagebox.showerror("Error", res.ERROR_MSG(e, ex))
				return 
	
	# Controls sub frames to show plot
	def Show_Plot(self, plot_data, property_dict):
		self.main_frame.close_plot()
		self.plot_data = plot_data
		self.property_dict = property_dict
		save_property_dict = [x for x in self.property_dict if x[0] not in  ['aplot_path', 'edf_path', 'anno_path']]
		fs.write_aplot(fs.Filepaths.TempAplotFile, self.plot_data, save_property_dict)
		self.main_frame.open_plot()

	# Controls sub frames to close plot
	def Close_File(self):
		self.main_frame.close_plot()
		self.plot_data = []
		self.property_dict = {}
	
	def popup(self, text):
		'''
		Creates pop-up shown by choosing help-menu items.
		'''
		# Create pop-up window
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

		# Binds
		toplevel.bind('<Escape>', lambda e: toplevel.destroy())
		toplevel.bind('<Return>', lambda e: toplevel.destroy())
		toplevel.bind('<Shift-C>', lambda e: toplevel.destroy())

		# Focus grab
		toplevel.lift()
		toplevel.focus_force()
		toplevel.grab_set()

		# Position
		x = self.winfo_x()
		y = self.winfo_y()
		toplevel.geometry("+%d+%d" % (x+5, y+55))

	# Main frame resource. It wraps plot frame and property frame unto same frame.
	class Main_Frame(Frame):
		# initilising
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

		# sub control: Show plot
		def open_plot(self):
			self.update_plot()
			self.plot_frame.grid()
			
			self.prop_frame.update_properties()
			self.prop_frame.grid()

		# sub control: Update plot
		def update_plot(self):
			plot = self.controller.plot_data
			self.controller.plot_figure = plot_results(*(list(plot) + [Figure(figsize=(res.ss_PLOT_WIDTH/res.ss_PLOT_DPI, res.ss_PLOT_HEIGHT/res.ss_PLOT_DPI), dpi=res.ss_PLOT_DPI)]))
			self.plot_frame.update_plot()

		# sub control: Close plot
		def close_plot(self):
			self.plot_frame.grid_remove()
			self.prop_frame.grid_remove()
		
		# Plot frame resource
		class Plot_Frame(Frame):
			# initilisation
			def __init__(self, master, controller):
				Frame.__init__(self, master,bg='white') # Super.__init__()
				self.controller = controller

				# Widget Packing
				self.plot = None

				# Grid
				self.update_plot()

			# sub sub control: update plot
			def update_plot(self):
				if self.controller.plot_figure: 
					subframe = Frame(self)

					canvas = FigureCanvasTkAgg(self.controller.plot_figure, subframe)
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
		
		# Property frame resource
		class Prop_Frame(Frame):
			# Initilisation
			def __init__(self, master, controller):
				Frame.__init__(self, master, bg='white') # Super.__init__()
				self.controller = controller
				self.properties = []
				self.update_properties()
			
			# sub sub control: Update properties
			def update_properties(self):
				if self.properties != None:
					for frame in self.properties:
						frame.grid_forget()

				w = res.ss_LABEL_WIDTH
				for k,v in self.controller.property_dict:
					subframe = self.__property(str(k), str(v), w, res.FONT)
					subframe.grid(sticky=NE)
					self.properties += [subframe]

			# Property sub resource
			def __property(self, key, val, width, font):
				subframe = Frame(self)
				Label(subframe, text=key, font=font, width=width).grid(row=0, column=0)
				Separator(subframe, orient=VERTICAL).grid(row=0, column=1, rowspan=3, sticky=NS)
				Label(subframe, text=val, font=font, width=width).grid(row=0, column=2)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=0, rowspan=3, sticky=EW)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=1, rowspan=3, sticky=EW)
				Separator(subframe, orient=HORIZONTAL).grid(row=1, column=2, rowspan=3, sticky=EW)
				return subframe

# Main method running the application
if __name__ == '__main__':
	App = AppUI()
	App.mainloop() # start GUI thread