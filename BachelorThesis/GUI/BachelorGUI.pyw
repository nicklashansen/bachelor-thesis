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
	def __init__(self):
		Tk.__init__(self)

		# Var
		self.plot_data = None
		global progbarThread
		progbarThread = None

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
		global progbarThread
		if not progbarThread:
			# Get File
			try:
				filepath = filedialog.askopenfilename(title='Choose PSG Recording File', filetypes=[(res.ff_FILETITLE_e,'*'+res.ff_FILETAG_e)])
				if not filepath or filepath == '':
					raise()
				# TODO: sleep stage anno file
			except Exception as e:
				return

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
						time.sleep(1.0/size) 
						# step out of 100%
						pb.step(100/size)

					self.plot_Data = None
					self.main_frame.open_plot(self.plot_data)
				except Exception as e:
					if progbarThread.getName() != 'close':
						self.Close_File()
				finally:
					b.grid_forget()
					pb.grid_forget()
					progbarThread = None
					self.unbind('<Escape>')

			def cancel():
				global progbarThread
				if progbarThread and progbarThread.is_alive() and progbarThread.getName() != 'cancel':
					progbarThread.setName('cancel') # Raise Flag

			# Close Current
			self.Close_File()

			# Progbar
			pb = Progressbar(self.main_frame)
			pb.grid(row=0, column=0)
			self.bind('<Escape>', lambda e: cancel())

			# Cancel Button
			b = Button(self.main_frame, text='Cancel')
			orig_color = b.cget("background")
			b.bind('<Button-1>', lambda event: cancel())
			b.bind('<Enter>', lambda e: b.configure(bg = 'SystemButtonHighlight'))
			b.bind('<Leave>', lambda e: b.configure(bg = orig_color))
			b.grid(row=0, column=1)
			
			# ProgbarThread
			progbarThread = threading.Thread(target=task, args=(filepath, pb, b))
			progbarThread.setName('start')
			progbarThread.start()

	# Open already formatted plots
	def Open_File(self):
		global progbarThread
		if not progbarThread:
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
		global progbarThread
		if progbarThread and progbarThread.is_alive() and progbarThread.getName() != 'close':
			progbarThread.setName('close') # Raise Flag
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
		def __init__(self, master, controller):
			Frame.__init__(self, master) # Super.__init__()
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
				Label(subframe, text='plot_menubar').grid()
				# Raw
				# Raw featues
				# Final features
				# arousals
				return subframe

			# main plt
			def __plot_main(self):
				subframe = Frame(self)
				Label(subframe, text='plot_main').grid()
				# Raw
				# Raw featues
				# Final features
				# arousals
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