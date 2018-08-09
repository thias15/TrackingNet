mode con: cols=100 lines=50

activate TrackingNet & python DownloadVideos.py "CSV_Files\Dataset_Test.csv" "Dataset\Test" --extract_frame --num_threads=16 & set /p temp="Hit enter to exit"