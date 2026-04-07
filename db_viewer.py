import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext

class DatabaseViewer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.root = tk.Tk()
        self.root.title("MindGuard AI - Database Viewer")
        self.root.geometry("1000x600")
        self.root.configure(bg='#0a1628')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: View Data
        self.view_frame = ttk.Frame(notebook)
        notebook.add(self.view_frame, text="📊 View Data")
        self.setup_view_tab()
        
        # Tab 2: SQL Query
        self.query_frame = ttk.Frame(notebook)
        notebook.add(self.query_frame, text="🔍 SQL Query")
        self.setup_query_tab()
        
        # Tab 3: Statistics
        self.stats_frame = ttk.Frame(notebook)
        notebook.add(self.stats_frame, text="📈 Statistics")
        self.setup_stats_tab()
        
        self.root.mainloop()
    
    def setup_view_tab(self):
        # Treeview for data
        self.tree = ttk.Treeview(self.view_frame)
        self.tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbars
        vsb = ttk.Scrollbar(self.view_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(self.view_frame, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x')
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Refresh button
        refresh_btn = tk.Button(self.view_frame, text="🔄 Refresh Data", 
                               command=self.load_data, bg='#40e0d0', fg='black')
        refresh_btn.pack(pady=5)
        
        self.load_data()
    
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM screenings ORDER BY created_at DESC", conn)
        conn.close()
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Set columns
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        # Configure columns
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Insert data
        for _, row in df.iterrows():
            values = [str(row[col])[:50] for col in df.columns]
            self.tree.insert("", "end", values=values)
    
    def setup_query_tab(self):
        # Text area for SQL query
        self.sql_text = scrolledtext.ScrolledText(self.query_frame, height=10, font=('Courier', 10))
        self.sql_text.pack(fill='x', padx=5, pady=5)
        self.sql_text.insert('1.0', "SELECT * FROM screenings WHERE risk_level = 'High';")
        
        # Execute button
        exec_btn = tk.Button(self.query_frame, text="▶ Execute Query", 
                            command=self.execute_query, bg='#40e0d0', fg='black')
        exec_btn.pack(pady=5)
        
        # Results tree
        self.result_tree = ttk.Treeview(self.query_frame)
        self.result_tree.pack(fill='both', expand=True, padx=5, pady=5)
    
    def execute_query(self):
        query = self.sql_text.get('1.0', tk.END).strip()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Get results
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Clear previous results
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            if columns:
                self.result_tree["columns"] = columns
                self.result_tree["show"] = "headings"
                
                for col in columns:
                    self.result_tree.heading(col, text=col)
                    self.result_tree.column(col, width=100)
                
                for row in rows:
                    self.result_tree.insert("", "end", values=[str(val) for val in row])
                
                messagebox.showinfo("Success", f"Query returned {len(rows)} rows")
            else:
                messagebox.showinfo("Success", "Query executed successfully (no results)")
            
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def setup_stats_tab(self):
        conn = sqlite3.connect(self.db_path)
        
        # Total screenings
        total = pd.read_sql_query("SELECT COUNT(*) as count FROM screenings", conn)
        total_label = tk.Label(self.stats_frame, text=f"📊 Total Screenings: {total['count'][0]}", 
                               font=('Arial', 14, 'bold'), bg='#0a1628', fg='white')
        total_label.pack(pady=10)
        
        # Risk distribution
        risk_dist = pd.read_sql_query("SELECT risk_level, COUNT(*) as count FROM screenings GROUP BY risk_level", conn)
        
        # Create text widget for stats
        stats_text = scrolledtext.ScrolledText(self.stats_frame, height=15, font=('Courier', 10))
        stats_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        stats_text.insert(tk.END, "=" * 50 + "\n")
        stats_text.insert(tk.END, "RISK LEVEL DISTRIBUTION\n")
        stats_text.insert(tk.END, "=" * 50 + "\n\n")
        
        for _, row in risk_dist.iterrows():
            stats_text.insert(tk.END, f"{row['risk_level']:10} : {row['count']} screenings\n")
        
        # Additional stats
        avg_risk = pd.read_sql_query("SELECT AVG(risk_probability) as avg FROM screenings", conn)
        avg_age = pd.read_sql_query("SELECT AVG(age) as avg FROM screenings", conn)
        
        stats_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        stats_text.insert(tk.END, "AGGREGATE STATISTICS\n")
        stats_text.insert(tk.END, "=" * 50 + "\n\n")
        stats_text.insert(tk.END, f"Average Risk Score : {avg_risk['avg'][0]:.4f}\n")
        stats_text.insert(tk.END, f"Average Age        : {avg_age['avg'][0]:.1f}\n")
        
        conn.close()

# Run the viewer
if __name__ == "__main__":
    db_path = "mindguard.db"
    viewer = DatabaseViewer(db_path)