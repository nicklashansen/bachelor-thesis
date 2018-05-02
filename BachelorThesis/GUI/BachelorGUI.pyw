from tkinter import *
from tkinter.ttk import Progressbar

def BachelorGUI(w=800, h=800):		
		root = Tk()
		root.geometry('{0}x{1}'.format(w,h))

		View = AppUI(root)

		View.pack()
		root.mainloop()

class AppUI(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master, width=master.winfo_width(), height=master.winfo_height()) # Super.__init__()

		self.menubar = self.__topmenubar()
		self.master.config(menu=self.menubar)

		# Slave Widgets
		main_window = Main_window(self)

		# Pack Slaves
		main_window.pack(fill=BOTH)
			
	def __topmenubar(self):
		menubar = Menu(self)

		# Filemenu
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="New", command=None)
		filemenu.add_command(label="Open", command=None)
		filemenu.add_command(label="Export", command=None)
		filemenu.add_command(label="Close", command=None)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit)
		menubar.add_cascade(label="File", menu=filemenu)

		# Helpmenu
		ABOUT_TEXT = "About \n Copyright (C) Michael Kirkegaard, Nicklas Hansen."
		def aboutCMD():
			print(ABOUT_TEXT)
		def aboutPopup():
			toplevel = Toplevel()
			label = Label(toplevel, text=ABOUT_TEXT, height=0, width=100)
			label.pack()

		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About_cmd", command=aboutCMD)
		helpmenu.add_command(label="About", command=aboutPopup)
		menubar.add_cascade(label="Help", menu=helpmenu)

		return menubar

class Main_window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master) # Super.__init__()

		# Slaves Packing
		self.progbar().pack(side=BOTTOM)
		Plot_window(self).pack(fill=BOTH)

	# Open File
	# Validate File (signals are present)
	# Run analysis
	# Loading bar
	def progbar(self):
		progbar = Progressbar(self)
		progbar.start()
		progbar.pack()
		return progbar

class Plot_window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master) # Super.__init__()

		# Widget Packing
		self.plot_menubar().pack(fill=X, side=TOP)
		self.plot_main().pack(fill=X, side=TOP)
		self.plot_statistics().pack(fill=X, side=BOTTOM)

	# plot menus
	def plot_menubar(self):
		subframe = Frame(self)
		Label(subframe, text='plot_menubar').pack()
		# Raw
		# Raw featues
		# Final features
		# arousals
		return subframe

	# main plt
	def plot_main(self):
		subframe = Frame(self)
		Label(subframe, text='plot_main').pack()
		# Raw
		# Raw featues
		# Final features
		# arousals
		return subframe

	# statitics
	def plot_statistics(self):
		subframe = Frame(self)
		Label(subframe, text='plot_statistics').pack()
		# Raw
		# Raw featues
		# Final features
		# arousals
		return subframe

if __name__ == '__main__':
	BachelorGUI()