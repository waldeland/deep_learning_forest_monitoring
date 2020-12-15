import argparse


if __name__=='__main__':

    p = argparse.ArgumentParser(description='Run clear cutting detection on all Sentinel-2 recordings between two dates')
    p.add_argument('-t', '--tileid', type=str,help='Sentinel-2 tile ID')
    p.add_argument('-p', '--tmp_path', type=str,help='path to tmp-files')
    p.add_argument('-s', '--start', type=str, help='Start date on format YYYYMMDD')
    p.add_argument('-e', '--end', type=str, help='End date on format YYYYMMDD')
    p.add_argument('-b', '--box', type=str, help='Coordinates for region of interest on format (y0,y1,x0,x1)')




