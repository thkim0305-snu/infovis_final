# InfoVis Final Project

1. Create a Python environment and install dependencies:

    ```bash
    conda create --name=pixplotml python=3.9
    conda activate pixplotml
    cd pixplot_server
    pip install -r requirements.txt
    ```
2. Download [output.tar](https://drive.google.com/file/d/1WMKdb_AhsgBzLEGB59-mgDg_Gan3yrcP/view?usp=drive_link) and unzip it inside pixplot_server:
    ```bash
    tar -xvf output.tar
    ```

3. Start a web server by running:

    ```bash
    python -m http.server 8600
    ```

    Open a browser to: `http://localhost:8600/output`.
