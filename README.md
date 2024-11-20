### README: Zotero RSS Manager

---

# Zotero RSS Manager

<img src="https://github.com/user-attachments/assets/187589e5-1a53-4407-9293-84a89c0878c3" alt="icon" width="200" />

<p style="font-size:14px; color:gray; font-style:italic; text-align:center;">
Science should be open, shared, and accessible to all.<br>
Research should be seamless, efficient, and empowering.
</p>

**Zotero RSS Manager** is a tool designed to enhance the use of Zotero by enabling better management of RSS feeds. It allows you to organize, score, and add RSS feed content to specific Zotero library folders efficiently. Using NLP (Natural Language Processing), it ranks RSS content based on relevance to your existing Zotero library folders, helping streamline your workflow. 

---

## Features

1. **Database Self-Inspection**:
   - Detect and remove invalid items in your Zotero database.
2. **RSS Feed Management**:
   - Browse and scan RSS feeds directly from your Zotero library.
3. **Content Scoring**:
   - Use NLP to compare RSS content with existing folder profiles and rank them by relevance.
4. **Article Management**:
   - Select and add top-scoring articles to specific Zotero folders.
5. **User-Friendly Interface**:
   - GUI elements like list boxes and buttons for folder selection and article confirmation.

---

## Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/YihanXie/ZoteroRSSManager.git
   cd ZoteroRSSManager
   ```

2. Install required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have **Zotero** installed and know the location of its database file (`zotero.sqlite`).

4. For running the standalone executable:
   - Download the pre-built `.exe` from the releases section, or
   - Build it yourself using:
     ```bash
     pyinstaller --clean --onefile --icon="icon.ico" --name "Zotero RSS Manager" --hidden-import dotenv --hidden-import email_validator --hidden-import importlib_resources.trees --hidden-import MySQLdb --hidden-import notebook.services.shutdown --hidden-import mx.DateTime --exclude-module egenix-mx-base --exclude-module pydantic.experimental --runtime-hook=mock_imports.py main.py
     ```

---

## Usage

### Adding RSS Feeds to Zotero

Before using this tool, you need to add RSS feeds to your Zotero library:
1. Open Zotero.
2. Navigate to the **Library** pane.
3. Right-click and select **New Feed**.
4. Enter the feed URL and a name for the feed.
5. Zotero will automatically fetch articles from the RSS feed.

### Using Zotero RSS Manager

1. **Run the Application**:
   - Run the Python script:
     ```bash
     python main.py
     ```
   - Or execute the standalone `.exe` file:
     ```bash
     ./Zotero RSS Manager.exe
     ```

2. **Select the Zotero Database**:
   - Browse and select your Zotero database file (e.g., `zotero.sqlite`).

3. **Database Self-Inspection**:
   - On startup, the tool inspects the Zotero database for invalid items.
   - If invalid items are found, you can choose to remove them.

4. **Folder and RSS Selection**:
   - Select Zotero library folders and RSS feeds for processing using the GUI.
   - Specify the number of top articles to display (default is 5).

5. **Scan and Rank Articles**:
   - The tool scans selected RSS feeds, processes abstracts using NLP, and ranks articles based on their relevance to the selected folders.

6. **Review and Select Articles**:
   - Review top-ranked articles in a GUI.
   - Select articles to add to your Zotero folders.

7. **Add Articles to Zotero**:
   - Selected articles are added to the specified Zotero folders.
   - Duplicates are avoided, and metadata is preserved.

---

## FAQ

### What is the Zotero database file, and where can I find it?

The Zotero database (`zotero.sqlite`) contains all your Zotero data, including library items, RSS feeds, and folders. You can find it in:
- **Windows**: `C:\Users\<Your Username>\Zotero\zotero.sqlite`
- **macOS**: `~/Zotero/zotero.sqlite`
- **Linux**: `~/.zotero/zotero.sqlite`

### How does the RSS scoring work?

1. Each Zotero library folder is analyzed to create a "profile" based on existing content.
2. RSS abstracts are processed using SpaCy's NLP model to extract key information.
3. TF-IDF and cosine similarity are used to score the relevance of RSS content to each folder.

### Can I use my custom NLP model?

Yes! Modify the `initialize_nlp()` function to load your custom SpaCy model or another NLP pipeline.

---

## License

This project is licensed under the **GNU General Public License v3**. See the full license at [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Contact

For issues, suggestions, or contributions, please contact:

**Yihan Xie**  
Email: [Yihan.Xie1@pennmedicine.upenn.edu](mailto:Yihan.Xie1@pennmedicine.upenn.edu)

Feel free to create a GitHub issue for technical problems.
