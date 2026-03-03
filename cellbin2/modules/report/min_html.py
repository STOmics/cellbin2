# how to install package:
# pip install csscompressor 
# pip install jsmin
# pip install beautifulsoup4

import re
import sys
import base64
import io
from PIL import Image

from bs4 import BeautifulSoup
# sys.path.append("c:\\users\\zhanghaorui\\appdata\\local\\programs\\python\\python38\\lib\\site-packages")
from csscompressor import compress
from jsmin import jsmin

# infile = sys.argv[1]
# outfile = 'StereoReport_v8.2.0_merge.html'

def minify_html(html_string):
  # html_string = re.sub(r'\/\/.*', '', html_string) # space, newline
  html_string = re.sub(r'\s+', ' ', html_string) # space, newline space、enter
  html_string = re.sub(r'>\s+<', '><', html_string) #between labels
  html_string = re.sub(r'=\s*"(.*?)"', '="\g<1>"', html_string) #space between attributes 
  html_string = re.sub(r'<!--(.*?)-->', '', html_string) #comment  
  html_string = re.sub(r'console.log\(.*?\);', '', html_string) #  
  html_string = re.sub(r'\s*([{};=])\s*', '\g<1>', html_string) #comment  
  html_string = re.sub(r'([:])\s*', '\g<1>', html_string) #comment  
  html_string = re.sub(r'\s+([><])\s+', '\g<1>', html_string) #comment  
  return html_string
# only min html:
# with open(infile, 'r', encoding = 'utf-8') as file:
#   html_content = file.read() 
# with open(outfile, 'w', encoding = 'utf-8') as f:
#   f.write(minify_html(html_content))
  
# use htmlmin: 
# import htmlmin #import minify
# outfile2 = f'min2_{infile}'
# with open(outfile2, 'w') as f:
#   f.write(htmlmin.minify(html_content, remove_empty_space = True))

def convert_png_to_base64(html):
    pattern = r'"([./\\]?[\\/.\w-]+\.png\b)"'
    # print(re.findall(pattern, html))
    def replace_func(match):
        src = match.group(1)
        # with open(src, "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # return '"data:image/png;base64,{}"'.format(encoded_string)

        image = Image.open(src)
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP')
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return '"data:image/webp;base64,{}"'.format(encoded_string)

    converted_html = re.sub(pattern, replace_func, html)
    return converted_html


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return 'data:image/png;base64,' + encoded_string
  
def operat_html(html_path,outfile):
    # read HTML file read HTML file
    with open(html_path, 'r', encoding = 'utf-8') as file:
        html_content = file.read()
    html_content = minify_html(html_content)
    # parse HTML file 
    soup = BeautifulSoup(html_content, 'html.parser')

    # get all script tags 
    script_tags = soup.find_all('script')
    # get all link tags
    link_tags = soup.find_all('link')
    # get all img tags 
    img_tags = soup.find_all('img')
    # iterate through script tags
    for script_tag in script_tags:
        # get the src attribute and content from script tags 
        src = script_tag.get('src')
        content = script_tag.string

        # if src attribute exist, fetch and compress corresponding local JS file 
        if src:
            with open(src, 'r', encoding = 'utf-8') as js_file:
                content = js_file.read()
            del script_tag["src"]

            # compress the JS file content 
            if 'module' in src:
                content = minify_html(content)
            if 'result.js' in src:
                content = jsmin(content)

        # replace the script tag content with compressed JS file content 
            script_tag.string = content


    # iterate through link tags 
    for link_tag in link_tags:
        # get the href attribute from link tag 
        href = link_tag.get('href')

        # if href attribute exist, fetch corresponding CSS file content
        if href and href.endswith('css'):
            with open(href, 'r', encoding = 'utf-8') as file:
              css_content = file.read()

            # compress CSS file content
            compressed_css_content = compress(css_content)

            # creat 'style' tag and assign compressed CSS file content to 'string' attribute in 'style' tag 
            style_tag = soup.new_tag('style')
            style_tag.string = compressed_css_content

            # replace link tag with style tag
            link_tag.replace_with(style_tag)
        else:
            link_tag['href'] = image_to_base64(link_tag['href'])

    # iterate through img tags 
    for img_tag in img_tags:
        # get the src attribute from img tags 
        src = img_tag.get('src')
        # if href attribute exist, fetch the corresponding CSS file content 
        if src:
            img_tag['src'] = image_to_base64(img_tag['src'])



    html_content = convert_png_to_base64(str(soup))

    # write the replaced HTML in new file 
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write(html_content)
          
# main(infile)