'''
Copyright (c) 2024 Yihan Xie

This project is licensed under the GNU General Public License v3.

You are free to:
- Use, copy, and distribute this software for personal or educational purposes.
- Modify and redistribute modified versions under the same license.

Restrictions:
- Commercial use of this software is strictly prohibited without prior written permission from the copyright holder.

For more details, see the full license text at https://www.gnu.org/licenses/gpl-3.0.en.html.

Contact: Yihan.Xie1@pennmedicine.upenn.edu
'''

#### handle the hidden import error

'''
# Mock jaraco and backports modules
sys.modules["jaraco"] = types.ModuleType("jaraco")
sys.modules["jaraco.context"] = types.ModuleType("context")
sys.modules["jaraco.text"] = types.ModuleType("text")
sys.modules["backports"] = types.ModuleType("backports")
sys.modules["backports.tarfile"] = types.ModuleType("tarfile")

if "egenix-mx-base" not in sys.modules:
    sys.modules["mx.DateTime"] = type("Mock", (), {"DateTime": None})

try:
    import importlib_resources.trees
except ImportError:
    pass
try:
    from notebook.services import shutdown
except ImportError:
    pass

try:
    import pkg_resources._vendor.jaraco.functools
    import pkg_resources._vendor.jaraco.context
    import pkg_resources._vendor.jaraco.text
    import pkg_resources.extern
except ImportError:
    pass

# Mock backports.tarfile to avoid import issues
if "backports" in sys.modules:
    sys.modules["backports.tarfile"] = None
'''
####


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
'''
########
import os
import sys
import spacy
# Set TensorFlow to use only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import tempfile

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Set TensorFlow to use only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Handle SpaCy model path
if getattr(sys, 'frozen', False):  # Running from PyInstaller bundle
    try:
        # Extract en_core_web_sm to a temporary directory
        model_source_path = os.path.join(sys._MEIPASS, "en_core_web_sm")
        model_temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_temp_dir, "en_core_web_sm")

        if not os.path.exists(model_path):
            shutil.copytree(model_source_path, model_path)
        print(f"Extracted model to temporary path: {model_path}")
    except Exception as e:
        print(f"Error preparing model path: {e}")
        sys.exit(1)
else:  # Running in a normal Python environment
    model_path = "en_core_web_sm"

# Load the SpaCy model
try:
    nlp = spacy.load(model_path)
    print("SpaCy model loaded successfully!")
except Exception as e:
    print(f"Error loading SpaCy model: {e}")
    print(f"Expected model path: {model_path}")
    sys.exit(1)
    
#################
'''
import os
import sys
import tempfile
import shutil
import spacy
from pathlib import Path
from spacy.util import load_model_from_path


# Debugging helper to list directory contents
def list_directory(path, description=""):
    print(f"\nContents of {description} ({path}):")
    for root, dirs, files in os.walk(path):
        print(f"{root}:")
        for d in dirs:
            print(f"  DIR: {d}")
        for f in files:
            print(f"  FILE: {f}")


# Function to dynamically find the spaCy model directory
def find_spacy_model(base_path):
    """Recursively find the spaCy model directory with meta.json and config.cfg."""
    for root, dirs, files in os.walk(base_path):
        if "meta.json" in files and "config.cfg" in files:
            return Path(root)  # Ensure it returns a Path object
    raise FileNotFoundError(f"Valid spaCy model not found in {base_path}")


# Simulate PyInstaller environment for testing
if "PYINSTALLER_TEST" in os.environ:
    sys.frozen = True
    sys._MEIPASS = os.environ["PYINSTALLER_TEST"]
    print(f"Simulated _MEIPASS directory: {sys._MEIPASS}")

# Set TensorFlow to use only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Handle SpaCy model path
if getattr(sys, 'frozen', False):  # Running from PyInstaller bundle
    try:
        # Log _MEIPASS directory for debugging
        print(f"_MEIPASS directory: {sys._MEIPASS}")
        list_directory(sys._MEIPASS, "_MEIPASS")

        # Locate the model in the PyInstaller bundle
        model_source_path = Path(sys._MEIPASS) / "en_core_web_sm"
        if not model_source_path.exists():
            raise FileNotFoundError(f"Model source path not found in _MEIPASS: {model_source_path}")

        # Extract to a temporary directory
        model_temp_dir = Path(tempfile.mkdtemp())
        extracted_model_path = model_temp_dir / "en_core_web_sm"
        shutil.copytree(model_source_path, extracted_model_path)
        print(f"Extracted model to temporary path: {extracted_model_path}")

        # Dynamically find the model path
        model_path = find_spacy_model(extracted_model_path)
        print(f"Located spaCy model directory: {model_path}")

    except Exception as e:
        print(f"Error preparing model path: {e}")
        sys.exit(1)
else:  # Normal Python environment
    model_path = Path("en_core_web_sm")  # Ensure it's a Path object

# Load the SpaCy model
try:
    print(f"Attempting to load SpaCy model from path: {model_path}")

    # Use `load_model_from_path` to load directly from the path
    nlp = load_model_from_path(model_path)
    print("SpaCy model loaded successfully!")

except Exception as e:
    print(f"Error loading SpaCy model: {e}")
    sys.exit(1)


#################
import sqlite3
from tkinter import filedialog, Listbox, Button, MULTIPLE
import time
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Toplevel
from spacy.tokens import Doc
import webbrowser
import tkinter as tk
from tkinter import Listbox, MULTIPLE, Button
from textwrap import wrap

import itertools

# Progress Printer Function
def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Connect to Zotero database
def connect_zotero_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print_progress(f"[ERROR] SQLite connection failed: {e}")
        return None

# Get Zotero folders
def get_zotero_folders(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT collectionName FROM collections")
    library_folders = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT name FROM feeds")
    rss_folders = [row[0] for row in cursor.fetchall()]
    return library_folders, rss_folders

# Remove invalid items from the database
def inspect_database_for_invalid_items(conn):
    """
    Inspect the database for invalid items.

    Parameters:
        conn (sqlite3.Connection): SQLite connection to the Zotero database.

    Returns:
        list: A list of invalid item IDs.
    """
    invalid_items = []
    try:
        cursor = conn.cursor()
        query = """
            SELECT itemID, key FROM items
            WHERE key LIKE 'newkey%' AND NOT EXISTS (
                SELECT 1 FROM itemData WHERE itemID = items.itemID
            )
        """
        cursor.execute(query)
        invalid_items = [row[0] for row in cursor.fetchall()]
        print_progress(f"[DEBUG] Found {len(invalid_items)} invalid items.")
    except sqlite3.Error as e:
        print_progress(f"[ERROR] Failed to inspect database: {e}")
    return invalid_items

def remove_invalid_items(conn, invalid_items):
    """
    Remove invalid items from the database.

    Parameters:
        conn (sqlite3.Connection): SQLite connection to the Zotero database.
        invalid_items (list): List of invalid item IDs.

    Returns:
        None
    """
    try:
        cursor = conn.cursor()
        cursor.executemany("DELETE FROM items WHERE itemID = ?", [(item_id,) for item_id in invalid_items])
        conn.commit()
        print_progress(f"[INFO] Removed {len(invalid_items)} invalid items from the database.")
    except sqlite3.Error as e:
        print_progress(f"[ERROR] Failed to remove invalid items: {e}")


def prompt_folder_selection_with_top_n(library_folders, rss_folders, conn):
    """
    Prompt user to select library and RSS folders with an additional button
    for database self-inspection. Uses preloaded GIF frames for animation.

    Parameters:
        library_folders (list): List of library folder names.
        rss_folders (list): List of RSS feed folder names.
        conn (sqlite3.Connection): SQLite connection to the Zotero database.
        gif_frames (list): Preloaded GIF frames.
        gif_frames_cycle (itertools.cycle): Cycle iterator for the GIF frames.

    Returns:
        tuple: Selected library folders, selected RSS feed folders, number of top articles.
    """
    selected_library = []
    selected_rss = []
    num_top_articles = 5  # Default value

    def on_confirm():
        """Handle confirmation of selected folders and number of articles."""
        nonlocal selected_library, selected_rss, num_top_articles
        selected_library = [library_listbox.get(i) for i in library_listbox.curselection()]
        selected_rss = [rss_listbox.get(i) for i in rss_listbox.curselection()]
        try:
            num_top_articles = int(num_articles_entry.get())
        except ValueError:
            num_top_articles = 5  # Default value if input is invalid
        print_progress(f"Confirmed selection: {selected_library} (Library), {selected_rss} (RSS), Top N: {num_top_articles}")
        root.quit()
        root.destroy()

    def database_self_inspection():
        """Perform self-inspection for invalid items in the database."""
        invalid_items = inspect_database_for_invalid_items(conn)
        if invalid_items:
            response = tk.messagebox.askyesno(
                "Database Self-Inspection",
                f"Found {len(invalid_items)} invalid items. Do you want to remove them?"
            )
            if response:
                remove_invalid_items(conn, invalid_items)
                print_progress("[INFO] Invalid items removed successfully.")
                tk.messagebox.showinfo("Database Self-Inspection", "Invalid items have been removed.")
            else:
                print_progress("[INFO] User chose not to remove invalid items.")
        else:
            tk.messagebox.showinfo("Database Self-Inspection", "No invalid items found.")
            print_progress("[INFO] No invalid items found during self-inspection.")


    root = tk.Tk()
    print_progress("[DEBUG] Tkinter root window initialized.")

    root.title("Zotero RSS Manager")
    root.geometry("800x700")

    # Create a frame for the title and image
    title_frame = tk.Frame(root)
    title_frame.pack(fill="x", pady=10)
    print_progress("[DEBUG] Title frame created.")

    title_label = tk.Label(title_frame, text="Zotero RSS Manager", font=("Arial", 18, "bold"))
    title_label.pack(side="left", padx=10)
    print_progress("[DEBUG] Title label added to title frame.")

    # Other UI elements...
    tk.Label(root, text="Select Library Folders:").pack()
    library_listbox = Listbox(root, selectmode=MULTIPLE, height=15, width=50, exportselection=False)
    library_listbox.pack()
    for folder in library_folders:
        library_listbox.insert(tk.END, folder)
    print_progress("[DEBUG] Library folders listbox populated.")

    tk.Label(root, text="Select RSS Feed Folders:").pack()
    rss_listbox = Listbox(root, selectmode=MULTIPLE, height=15, width=50, exportselection=False)
    rss_listbox.pack()
    for folder in rss_folders:
        rss_listbox.insert(tk.END, folder)
    print_progress("[DEBUG] RSS folders listbox populated.")

    tk.Label(root, text="Number of Top Articles to Display (Default: 5):").pack()
    num_articles_entry = tk.Entry(root, width=5)
    num_articles_entry.insert(0, "5")
    num_articles_entry.pack()
    print_progress("[DEBUG] Number of top articles entry created.")

    Button(root, text="Confirm Selection", command=on_confirm).pack(pady=5)
    print_progress("[DEBUG] Confirm button added.")

    Button(root, text="Database Self-Inspection", command=database_self_inspection).pack(pady=5)
    print_progress("[DEBUG] Database self-inspection button added.")

    # Add copyright text at the bottom
    copyright_label = tk.Label(root, text="Copyright (c) 2024 Yihan Xie", font=("Arial", 10, "italic"), fg="gray")
    copyright_label.pack(side="bottom", pady=10)
    print_progress("[DEBUG] Copyright label added.")

    print_progress("[DEBUG] Tkinter selection window is running...")
    root.mainloop()
    print_progress("[DEBUG] Tkinter selection window closed.")

    return selected_library, selected_rss, num_top_articles



# Extract folder profiles
def extract_folder_profile(conn, selected_library):
    cursor = conn.cursor()
    folder_profiles = {}
    for folder in selected_library:
        query = """
        SELECT itemDataValues.value
        FROM collectionItems
        JOIN items ON collectionItems.itemID = items.itemID
        JOIN itemData ON items.itemID = itemData.itemID
        JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
        WHERE collectionItems.collectionID = (
            SELECT collectionID FROM collections WHERE collectionName = ?
        )
        """
        cursor.execute(query, (folder,))
        values = cursor.fetchall()
        profile = " ".join([value[0] for value in values if value[0]])
        folder_profiles[folder] = profile
    return folder_profiles

def scan_rss_feeds(conn, selected_rss, nlp):
    """
    Scans selected RSS feeds for content.
    :param conn: SQLite connection to the Zotero database.
    :param selected_rss: List of selected RSS feed names.
    :param nlp: Loaded spaCy language model.
    :return: List of dictionaries with RSS feed entries.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name, url FROM feeds")
    rss_feeds = cursor.fetchall()
    total_feeds = len(rss_feeds)
    print_progress(f"Found {total_feeds} RSS feeds in Zotero.")

    new_content = []
    skipped_entries = 0

    for feed_idx, (feed_name, feed_url) in enumerate(rss_feeds, start=1):
        if feed_name not in selected_rss:
            print_progress(f"Skipping feed: {feed_name}")
            continue

        print_progress(f"Scanning RSS feed {feed_idx}/{total_feeds}: {feed_name}")
        parsed_feed = feedparser.parse(feed_url)
        total_entries = len(parsed_feed.entries)
        print_progress(f"Found {total_entries} entries in feed: {feed_name}")

        for entry_idx, entry in enumerate(parsed_feed.entries, start=1):
            title = entry.get("title", "No Title")
            abstract = entry.get("summary", None)  # Abstract may not exist
            pdf_url = entry.get("link", None)

            if not abstract:
                skipped_entries += 1
                print_progress(f"Skipping RSS entry {entry_idx}/{total_entries}: {title} (No Abstract Found)")
                continue

            print_progress(f"Processing RSS entry {entry_idx}/{total_entries}: {title}")

            try:
                # Safeguard for abstract being a non-string
                if not isinstance(abstract, str):
                    print_progress(f"Invalid abstract format for entry {entry_idx}: {abstract}")
                    skipped_entries += 1
                    continue

                # Process abstract using spaCy
                doc = nlp(abstract)
                summary = doc._.extractive_summary if hasattr(doc._, 'extractive_summary') else abstract[:200]

                new_content.append({
                    'feed': feed_name,
                    'title': title,
                    'summary': summary,
                    'url': pdf_url
                })
            except Exception as e:
                print_progress(f"[ERROR] Failed to process entry {entry_idx}/{total_entries}: {title}. Error: {e}")
                skipped_entries += 1
                continue

    print_progress(f"RSS feed scanning completed. Skipped {skipped_entries} entries.")
    return new_content



# Score RSS papers
def score_rss_papers(folder_profiles, rss_content):
    vectorizer = TfidfVectorizer()
    scores = {folder: [] for folder in folder_profiles}
    for folder, profile_text in folder_profiles.items():
        documents = [profile_text] + [paper['summary'] for paper in rss_content]
        tfidf_matrix = vectorizer.fit_transform(documents)
        folder_vector = tfidf_matrix[0]
        rss_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(folder_vector, rss_vectors)

        for i, paper in enumerate(rss_content):
            title = paper.get('title', 'No Title')
            score = similarities[0, i]
            url = paper.get('url', 'No URL')
            date = paper.get('date', 'Unknown')  # Provide a default value for missing date

            scores[folder].append((title, score, url, date))
        scores[folder] = sorted(scores[folder], key=lambda x: x[1], reverse=True)
    return scores


# Summarize top papers
def summarize_top_papers(scores, top_n=5):
     # For wrapping long titles

    print("=== Summarized Top Papers ===")
    for folder, papers in scores.items():
        print(f"\nTop {top_n} Papers for Folder: {folder}")

        # Debugging: Check the structure of the scores for the folder
        print(f"[DEBUG] Processing folder '{folder}' with {len(papers)} papers.")
        for idx, paper in enumerate(papers[:top_n]):
            if len(paper) < 4:
                print(f"[ERROR] Paper {idx} in folder '{folder}' has an invalid structure: {paper}")
                continue  # Skip invalid entries

            # Extract values for each paper
            title, score, url, date = paper

            # Debugging: Confirm correct extraction of score and fields
            print(f"[DEBUG] Paper {idx} details: Score={score}, Title={title[:50]}, Date={date}, URL={url}")

            # Handle missing date and URL gracefully
            formatted_date = date if date else "Unknown"
            formatted_url = url if url else "No URL available"

            # Wrap the title for better readability
            wrapped_title = "\n".join(wrap(title, width=80))

            # Print the information with relevance score first
            print(f"- Relevance Score: {score:.2f}")
            print(f"  Title: {wrapped_title}")
            #print(f"  Date: {formatted_date}")
            print(f"  URL: {formatted_url}\n")





# Display top articles for selection

def display_top_articles(scores):
    """
    Display a GUI for users to select top articles from different folders.

    Parameters:
        scores (dict): A dictionary where keys are folder names and values are lists of articles.
                       Each article is a tuple (title, score, url, date).

    Returns:
        list: A list of selected articles.
    """
    selected_articles = []  # Track selected articles
    selection_states = {}  # Track selection states by folder and index

    def open_link(url):
        """Open the URL in a web browser."""
        if url and url != "No URL available":
            webbrowser.open(url)
        else:
            print_progress("[ERROR] No valid URL to open.")

    def record_click(event, folder, idx, lbl):
        """Handle clicks to toggle article selection and update UI."""
        if selection_states[folder][idx]:
            selection_states[folder][idx] = False
            lbl.config(bg="SystemButtonFace")  # Reset to default color
            selected_articles.remove(scores[folder][idx])
            print_progress(f"[DEBUG] Article '{scores[folder][idx][0]}' deselected.")
        else:
            selection_states[folder][idx] = True
            lbl.config(bg="lightblue")  # Highlight selected article
            selected_articles.append(scores[folder][idx])
            print_progress(f"[DEBUG] Article '{scores[folder][idx][0]}' selected.")

    def on_confirm():
        """Capture the final selection and close the window."""
        print_progress(f"[DEBUG] Final selected articles: {selected_articles}")
        root.quit()
        root.destroy()

    def show_tooltip(event, title):
        """Display a tooltip showing the full title."""
        tooltip.title("Tooltip")
        tooltip_label.config(text=title)
        tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        tooltip.deiconify()  # Show the tooltip
        print_progress(f"[DEBUG] Tooltip displayed for '{title}' at ({event.x_root}, {event.y_root})")

    def hide_tooltip(event):
        """Hide the tooltip."""
        tooltip.withdraw()
        print_progress("[DEBUG] Tooltip hidden")

    root = tk.Tk()
    root.title("Select Articles for Library Folders")
    root.geometry("800x600")  # Adjusted for better layout

    article_frames = {}

    # Tooltip setup
    tooltip = Toplevel(root)
    tooltip.withdraw()  # Hide the tooltip initially
    tooltip.overrideredirect(True)  # Remove window decorations
    tooltip_label = tk.Label(tooltip, text="", bg="yellow", wraplength=400, justify="left", anchor="w")
    tooltip_label.pack()

    for folder, articles in scores.items():
        frame = tk.LabelFrame(root, text=folder, padx=5, pady=5)
        frame.pack(fill="both", expand="yes", padx=10, pady=10)

        article_frame = tk.Frame(frame)
        article_frame.pack(fill="both", expand="yes")

        article_frames[folder] = article_frame
        selection_states[folder] = [False] * len(articles)  # Initialize all selection states as False

        for idx, (title, score, url, date) in enumerate(articles):
            # Truncated article information
            truncated_title = title if len(title) <= 50 else title[:50] + "..."
            article_info = f"Score: {score:.2f} | {truncated_title} | Date: {date}"

            lbl = tk.Label(article_frame, text=article_info, anchor="w", wraplength=700, bg="SystemButtonFace")
            lbl.grid(row=idx, column=0, sticky="w", padx=5, pady=2)

            # Hover behavior for tooltip
            lbl.bind("<Enter>", lambda e, t=title: show_tooltip(e, t))
            lbl.bind("<Leave>", hide_tooltip)

            # Click behavior to toggle selection and color
            lbl.bind("<Button-1>", lambda e, f=folder, i=idx, l=lbl: record_click(e, f, i, l))

            print_progress(f"[DEBUG] Added label for article: {title[:50]}...")

            # Button to view article
            btn = tk.Button(article_frame, text="View Article", command=lambda u=url: open_link(u), width=15)
            btn.grid(row=idx, column=1, padx=5, pady=2)

    tk.Button(root, text="Confirm Selection", command=on_confirm).pack(pady=10)

    print_progress("Tkinter article selection window is running...")
    root.mainloop()
    print_progress("Tkinter article selection window closed.")

    return selected_articles



def print_progress(message):
    print(message)  # Replace with logging if needed


# Function to initialize spaCy model and set extensions
def initialize_nlp():
    print_progress("Loading spaCy language model...")

    #nlp = spacy.load("en_core_web_sm")
    print_progress("spaCy language model loaded successfully.")

    # Register the custom extension
    print_progress("Registering spaCy extensions...")
    if not Doc.has_extension("extractive_summary"):
        Doc.set_extension("extractive_summary", getter=extractive_summary)
    print_progress("SpaCy extensions registered successfully.")
    return nlp

# Extractive summary function for spaCy documents
def extractive_summary(doc, num_sentences=3):
    sentences = [sent.text for sent in doc.sents]
    top_sentences = sorted(sentences, key=len, reverse=True)[:num_sentences]
    return " ".join(top_sentences)

# Progress Printer Function
def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")





# Add selected articles to the specified Zotero folder
def add_to_library_folder(conn, articles, folder_name):
    """
    Add selected articles to the Zotero library and a specified folder.

    Parameters:
        conn (sqlite3.Connection): SQLite connection to the Zotero database.
        articles (list): List of articles to add. Each article is a tuple (title, score, url, date).
        folder_name (str): Name of the folder to add articles to.

    Returns:
        None
    """
    try:
        # Get collection ID and libraryID for the folder
        cursor = conn.cursor()
        cursor.execute("SELECT collectionID, libraryID FROM collections WHERE collectionName = ?", (folder_name,))
        result = cursor.fetchone()

        if not result:
            print_progress(f"[ERROR] Folder '{folder_name}' does not exist.")
            return

        collection_id, target_library_id = result
        print_progress(f"[DEBUG] Folder '{folder_name}' has collectionID: {collection_id} and libraryID: {target_library_id}")

        def get_or_create_value(value):
            """Get the valueID for a metadata value, or create it if it doesn't exist."""
            cursor.execute("SELECT valueID FROM itemDataValues WHERE value = ?", (value,))
            result = cursor.fetchone()
            if result:
                return result[0]  # Existing valueID
            cursor.execute("INSERT INTO itemDataValues (value) VALUES (?)", (value,))
            return cursor.lastrowid  # New valueID

        for article in articles:
            title, score, url, date = article

            # Find itemID and libraryID for the article using its title
            cursor.execute("""
                SELECT i.itemID, i.libraryID, i.key
                FROM items i
                JOIN itemData d ON i.itemID = d.itemID
                JOIN itemDataValues v ON d.valueID = v.valueID
                JOIN fields f ON d.fieldID = f.fieldID
                WHERE f.fieldName = 'title' AND v.value = ?
            """, (title,))
            item = cursor.fetchone()

            if item:
                item_id, item_library_id, item_key = item

                if item_library_id != target_library_id:
                    print_progress(
                        f"[WARNING] Article '{title}' belongs to a different library (libraryID {item_library_id}). Duplicating to target library."
                    )

                    # Duplicate the item into the target library
                    cursor.execute("SELECT MAX(itemID) FROM items")
                    max_item_id = cursor.fetchone()[0] or 0
                    new_item_id = max_item_id + 1

                    # Generate a unique key
                    new_key = f"{item_key}-{new_item_id}"

                    cursor.execute("""
                        INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
                        SELECT ?, itemTypeID, datetime('now'), datetime('now'), datetime('now'), ?, ?, version, synced
                        FROM items WHERE itemID = ?
                    """, (new_item_id, target_library_id, new_key, item_id))

                    # Duplicate item metadata
                    cursor.execute("""
                        INSERT INTO itemData (itemID, fieldID, valueID)
                        SELECT ?, fieldID, valueID FROM itemData WHERE itemID = ?
                    """, (new_item_id, item_id))

                    item_id = new_item_id  # Update to the duplicated item ID
                    print_progress(f"[INFO] Article '{title}' duplicated into libraryID {target_library_id} with new itemID {item_id} and key '{new_key}'.")

                # Check if the article is already in the folder
                cursor.execute(
                    "SELECT 1 FROM collectionItems WHERE collectionID = ? AND itemID = ?", (collection_id, item_id)
                )
                in_folder = cursor.fetchone()

                if in_folder:
                    print_progress(
                        f"[DEBUG] Article '{title}' already exists in folder '{folder_name}' with itemID {item_id}."
                    )
                    continue

                # Add the article to the folder
                cursor.execute(
                    """
                    INSERT INTO collectionItems (collectionID, itemID, orderIndex)
                    VALUES (?, ?, COALESCE((SELECT MAX(orderIndex) + 1 FROM collectionItems WHERE collectionID = ?), 0))
                    """, (collection_id, item_id, collection_id)
                )
                print_progress(f"[INFO] Article '{title}' added to folder '{folder_name}'.")
            else:
                print_progress(f"[INFO] Article '{title}' not found in library. Adding as a new item.")

                # Add new article to the library with the correct libraryID
                cursor.execute("SELECT MAX(itemID) FROM items")
                max_item_id = cursor.fetchone()[0] or 0
                new_item_id = max_item_id + 1

                # Generate a unique key
                new_key = f"newkey-{new_item_id}"

                cursor.execute("""
                    INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
                    VALUES (?, ?, datetime('now'), datetime('now'), datetime('now'), ?, ?, 1, 1)
                """, (new_item_id, 2, target_library_id, new_key))

                # Add metadata
                title_value_id = get_or_create_value(title)
                cursor.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)", (new_item_id, 1, title_value_id))

                if url:
                    url_value_id = get_or_create_value(url)
                    cursor.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)", (new_item_id, 13, url_value_id))

                if date:
                    date_value_id = get_or_create_value(date)
                    cursor.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)", (new_item_id, 5, date_value_id))

                print_progress(f"[INFO] New article '{title}' added to library with itemID {new_item_id} and key '{new_key}'.")

                # Add the new article to the folder
                cursor.execute(
                    """
                    INSERT INTO collectionItems (collectionID, itemID, orderIndex)
                    VALUES (?, ?, COALESCE((SELECT MAX(orderIndex) + 1 FROM collectionItems WHERE collectionID = ?), 0))
                    """, (collection_id, new_item_id, collection_id)
                )
                print_progress(f"[INFO] New article '{title}' added to folder '{folder_name}'.")

        # Commit the changes
        conn.commit()
        print_progress(f"[DEBUG] Successfully processed articles for folder '{folder_name}'.")

    except sqlite3.Error as e:
        print_progress(f"[ERROR] Database error while adding articles to folder '{folder_name}': {e}")

    except Exception as e:
        print_progress(f"[ERROR] Unexpected error: {e}")




def print_articles_in_folder(conn, folder_name):
    """
    Print all articles in the specified Zotero folder.
    """
    try:
        cursor = conn.cursor()
        query = """
            SELECT items.itemID, itemDataValues.value AS title
            FROM items
            JOIN collectionItems ON items.itemID = collectionItems.itemID
            JOIN collections ON collectionItems.collectionID = collections.collectionID
            JOIN itemData ON items.itemID = itemData.itemID
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            WHERE collections.collectionName = ?
              AND itemData.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
        """
        cursor.execute(query, (folder_name,))
        articles = cursor.fetchall()
        if articles:
            for article in articles:
                print(f"ID: {article[0]}, Title: {article[1]}")
        else:
            print(f"No articles found in folder '{folder_name}'.")
    except sqlite3.Error as e:
        print_progress(f"[ERROR] Failed to fetch articles from folder '{folder_name}': {e}")

def main():
    try:


        root = tk.Tk()
        root.withdraw()

        # Select the Zotero database file
        db_path = filedialog.askopenfilename(
            title="Select Zotero Database File",
            filetypes=[("SQLite Database", "*.sqlite")]
        )
        if not db_path:
            print_progress("[ERROR] No database file selected. Exiting.")
            return

        # Connect to the Zotero database
        conn = connect_zotero_db(db_path)
        if not conn:
            print_progress("[ERROR] Unable to connect to the database. Exiting.")
            return

        # Initialize the spaCy NLP model
        nlp = initialize_nlp()

        # Perform database self-inspection before proceeding
        invalid_items = inspect_database_for_invalid_items(conn)
        if invalid_items:
            print_progress(f"[WARNING] Found {len(invalid_items)} invalid items.")
            response = input("Do you want to remove invalid items? (y/n): ").strip().lower()
            if response == "y":
                remove_invalid_items(conn, invalid_items)
                print_progress("[INFO] Invalid items removed successfully.")
            else:
                print_progress("[INFO] Skipped removing invalid items.")

        # Fetch library and RSS folders
        library_folders, rss_folders = get_zotero_folders(conn)
        if not library_folders and not rss_folders:
            print_progress("[ERROR] No folders found in the database. Exiting.")
            return



        # Pass the preloaded GIF objects to the prompt function
        selected_library, selected_rss, num_top_articles = prompt_folder_selection_with_top_n(
            library_folders, rss_folders, conn
        )

        if not selected_library and not selected_rss:
            print_progress("[ERROR] No folders selected. Exiting.")
            return

        # Extract folder profiles
        folder_profiles = extract_folder_profile(conn, selected_library)

        # Scan selected RSS feeds
        rss_content = scan_rss_feeds(conn, selected_rss, nlp)
        if not rss_content:
            print_progress("[ERROR] No RSS content found. Exiting.")
            conn.close()
            return

        # Score the RSS papers for relevance
        scores = score_rss_papers(folder_profiles, rss_content)
        if not scores:
            print_progress("[ERROR] No scores generated. Exiting.")
            conn.close()
            return

        # Trim scores to the top N articles per folder
        for folder in scores:
            scores[folder] = scores[folder][:num_top_articles]

        # Display top articles for user selection
        selected_articles = display_top_articles(scores)

        # Add the selected articles to the corresponding library folders
        for folder in selected_library:
            print_progress(f"\n=== Articles in Folder '{folder}' BEFORE Adding New Articles ===")
            print_articles_in_folder(conn, folder)

            # Filter articles relevant to the current folder
            folder_articles = [
                article for article in selected_articles if any(
                    scored_article[0] == article[0] for scored_article in scores.get(folder, [])
                )
            ]

            if folder_articles:
                print_progress(f"Adding {len(folder_articles)} articles to folder '{folder}'...")
                add_to_library_folder(conn, folder_articles, folder)
            else:
                print_progress(f"No articles to add to folder '{folder}'.")

            print_progress(f"\n=== Articles in Folder '{folder}' AFTER Adding New Articles ===")
            print_articles_in_folder(conn, folder)

        conn.close()
        print_progress("Process completed successfully.")

    except Exception as e:
        print_progress(f"[ERROR] An exception occurred: {e}")


if __name__ == "__main__":
    main()