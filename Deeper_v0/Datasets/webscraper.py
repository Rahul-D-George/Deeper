import requests as rq
from bs4 import BeautifulSoup as BS
import os


headers = {
    'User-Agent': 'scraping for character icons :P'
}

all_names_req = rq.get(r"https://genshin-impact.fandom.com/wiki/Character/List")
chars_html = all_names_req.content
chars_soup = BS(chars_html, features="html.parser")
name_soup = chars_soup.find("table")
names = []

for row in name_soup.find_all("tr"):
    data_items = row.find_all("td")
    if len(data_items) > 0:
        name = str(data_items[0].find("a")["title"])
        if " " in name:
            name = name.replace(" ", "_")
        names.append(name)

dir_name = os.path.join(os.getcwd(), "../img_set")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
rel_path = os.getcwd() + "\\" + "img_set"
files = os.listdir(dir_name)

for name in names:
    if (name+".png") not in files:
        url = f"https://genshin-impact.fandom.com/wiki/{name}/Gallery"
        name_req = rq.get(url, headers)
        all_imgs = BS(name_req.content, features="html.parser").find_all("img")
        icon_url = ""
        for img in all_imgs:
            if img["alt"] == "Character Icon":
                icon_url = img["src"]
                break
        path = os.path.join(rel_path, name + ".png")
        img_req = rq.get(icon_url, headers)
        f = open(path, "wb")
        f.write(img_req.content)
        f.close()
        print(f"Added: {name}")
