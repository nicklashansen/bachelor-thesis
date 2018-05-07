from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
import time
import threading

FILETAG = '.aplot'

class AppUI(Tk):
	def __init__(self):
		Tk.__init__(self)

		# Slave Widgets 
		self.main_window = Main_Window(self)

		# Grid
		self.main_window.grid(sticky=N+E+S+W)

		# Menu
		self.menubar = self.__topmenubar()
		self.config(menu=self.menubar)

		# Shortcuts
		self.bind('<Shift-N>', lambda e: self.main_window.New_File())
		self.bind('<Shift-O>', lambda e: self.main_window.Open_File())
		self.bind('<Shift-S>', lambda e: self.main_window.Save_File())
		self.bind('<Shift-Enter>', lambda e: self.main_window.Close_File())
		self.bind('<Shift-Escape>', lambda e: self.quit())

		# Init
		self.mainloop()
			
	def __topmenubar(self):
		menubar = Menu(self)

		# Filemenu
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="New", command=self.main_window.New_File, accelerator='<Shift-N>')
		filemenu.add_command(label="Open", command=self.main_window.Open_File, accelerator='<Shift-O>')
		filemenu.add_command(label="Save", command=self.main_window.Save_File, accelerator='<Shift-S>')
		filemenu.add_command(label="Close", command=self.main_window.Close_File, accelerator='<Shift-Enter>')
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit, accelerator='<Shift-Esc>')
		menubar.add_cascade(label="File", menu=filemenu)

		# Helpmenu
		ABOUT_TEXT = [	"About",
						"-"*50,
						"Copyright (C) Michael Kirkegaard, Nicklas Hansen."
						]

		FORMAT_TEXT = [	"Format",
						"-"*50,
						"- PSG files must be in European Data Format (*.edf file extension) and must contain PPG and EKG signals.",
						"- Arousal Plots are saved with *.aplot file extension and must be created through this software."
						]
		
		COMMANDS_TEXT =	[	"Commands",
							"-"*50,
							"- New	    <Shift-N>:         Open a PSG file for analysis.",
							"- Open	    <Shift-O>:         Open an arousal plot file (i.e. an already analysed PSG file).",
							"- Save	    <Shift-S>:         Save an arousal plot file.",
							"- Close    <Shift-Enter> :    Close currently opened arousal plot file.",
							"- Exit	    <Shift-Escape>:    Close application."
							]

		def popup(text):
			# popup
			toplevel = Toplevel()
			# labels
			for i,s in enumerate(text):
				label = Label(toplevel, text=s, width=100, font='monospace 10', anchor=W if i >= 2 else None)
				label.grid(row=i, column=0)
			# Focus grab
			toplevel.bind('<Escape>', lambda e: toplevel.destroy())
			toplevel.lift()
			toplevel.focus_force()
			toplevel.grab_set()
			toplevel.grab_release()

		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About", command=lambda: popup(ABOUT_TEXT))
		helpmenu.add_command(label="Format", command=lambda: popup(FORMAT_TEXT))
		helpmenu.add_command(label="Commands", command=lambda: popup(COMMANDS_TEXT))
		menubar.add_cascade(label="Help", menu=helpmenu)

		return menubar

class Main_Window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master) # Super.__init__()

		self.plot_Data = None

		# Slaves
		self.plot_window = Plot_Window(self)
		self.prop_window = Prop_Window(self)

		# Grid
		self.plot_window.grid(row=0, column=0)
		self.prop_window.grid(row=1, column=0)

		# Default 
		self.Close_File()

	# New EDF File
	global progbarThread
	progbarThread = None
	def New_File(self):
		def task(filepath, pb, b):
			global progbarThread
			try:
				# Mockup file
				size = 1000
				for _ in range(size):
					# Soft Close
					if progbarThread.getName() in ['cancel','close']: # Shutdown Flags
						raise()
					# Do files and stuff
					time.sleep(0.001) 
					# step out of 100%
					pb.step(100/size)

				self.plot_Data = None
				self.__Open_Plot()
			except:
				if progbarThread.getName() != 'close':
					self.Close_File()
			finally:
				b.grid_forget()
				pb.grid_forget()
				progbarThread = None

		def cancel():
			global progbarThread
			if progbarThread and progbarThread.is_alive() and progbarThread.getName() != 'cancel':
				progbarThread.setName('cancel') # Raise Flag

		global progbarThread
		if not progbarThread:
			# Get File
			try:
				filepath = filedialog.askopenfilename(title='Choose PSG Recording File', filetypes=[('European Data Format','*.edf')])
				if not filepath or filepath == '':
					raise()
			except:
				return

			# Close Current
			self.Close_File()

			# Progbar
			pb = Progressbar(self)
			pb.grid(row=0, column=0)

			# Cancel Button
			b = Button(self, text='Cancel')
			b.bind('<Button-1>', lambda event: cancel())
			b.grid(row=0, column=1)
			
			# ProgbarThread
			progbarThread = threading.Thread(target=task, args=(filepath, pb, b))
			progbarThread.setName('start')
			progbarThread.start()
			

	# Open already formatted plots
	def Open_File(self):
		try:
			file = filedialog.askopenfile(title='Choose Arousal Plot file', filetypes=[('Arousal Plot','*.aplot')])
			self.plot_Data = None # pickle.dump(file)
			file.close()
		except:
			return
		self.__Open_Plot()

	# Show plotfile
	def __Open_Plot(self):
		#if self.plot_Data:
			self.plot_window.grid()
			self.prop_window.grid()
			# TODO: Plots n' stuff
			# TODO: Properties n' stuff
	
	# Save plotfile
	def Save_File(self):
		if self.plot_Data:
			file = filedialog.asksaveasfile(filetypes=[('Arousal Plot','*.aplot')])
			#pickle.dump(file, self.plot_Data)
			file.close()

	# Close/Cancel
	def Close_File(self):
		global progbarThread
		if progbarThread and progbarThread.is_alive() and progbarThread.getName() != 'close':
			progbarThread.setName('close') # Raise Flag
		self.plot_window.grid_remove()
		self.prop_window.grid_remove()
		self.plot_Data = None

class Plot_Window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master) # Super.__init__()

		# Widget Packing
		self.__plot_menubar().grid(row=0, column=0)
		self.__plot_main().grid(row=1, column=0)

	# plot menus
	def __plot_menubar(self):
		subframe = Frame(self)
		Label(subframe, text='plot_menubar').pack()
		# Raw
		# Raw featues
		# Final features
		# arousals
		return subframe

	# main plt
	def __plot_main(self):
		subframe = Frame(self)
		Label(subframe, text='plot_main').pack()
		# Raw
		# Raw featues
		# Final features
		# arousals
		return subframe

class Prop_Window(Frame):
	def __init__(self, master = None):
		Frame.__init__(self, master) # Super.__init__()

		# Widget slave packing
		self.__plot_properties().pack()
	
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