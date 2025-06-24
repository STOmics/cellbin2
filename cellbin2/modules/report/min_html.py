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
  # html_string = re.sub(r'\/\/.*', '', html_string) # 空格、换行
  html_string = re.sub(r'\s+', ' ', html_string) # 空格、换行
  html_string = re.sub(r'>\s+<', '><', html_string) #标签之间
  html_string = re.sub(r'=\s*"(.*?)"', '="\g<1>"', html_string) #属性之间空格
  html_string = re.sub(r'<!--(.*?)-->', '', html_string) #注释  
  html_string = re.sub(r'console.log\(.*?\);', '', html_string) #  
  html_string = re.sub(r'\s*([{};=])\s*', '\g<1>', html_string) #注释  
  html_string = re.sub(r'([:])\s*', '\g<1>', html_string) #注释  
  html_string = re.sub(r'\s+([><])\s+', '\g<1>', html_string) #注释  
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
    # 读取HTML文件
    with open(html_path, 'r', encoding = 'utf-8') as file:
        html_content = file.read()
    html_content = minify_html(html_content)
    # 解析HTML文件
    soup = BeautifulSoup(html_content, 'html.parser')

    # 获取所有的script标签
    script_tags = soup.find_all('script')
    # 获取所有的link标签
    link_tags = soup.find_all('link')
    # 获取所有的img标签
    img_tags = soup.find_all('img')
    # 遍历script标签
    for script_tag in script_tags:
        # 获取script标签中的src属性和内容
        src = script_tag.get('src')
        content = script_tag.string

        # 如果src属性存在，则读取对应的本地JS文件内容进行压缩
        if src:
            with open(src, 'r', encoding = 'utf-8') as js_file:
                content = js_file.read()
            del script_tag["src"]

            # 压缩JS文件内容
            if 'module' in src:
                content = minify_html(content)
            if 'result.js' in src:
                content = jsmin(content)

        # 替换script标签的内容为压缩后的JS文件内容
            script_tag.string = content


    # 遍历link标签
    for link_tag in link_tags:
        # 获取link标签中的href属性
        href = link_tag.get('href')

        # 如果href属性存在，则获取对应的CSS文件内容
        if href and href.endswith('css'):
            with open(href, 'r', encoding = 'utf-8') as file:
              css_content = file.read()

            # 压缩CSS文件内容
            compressed_css_content = compress(css_content)

            # 创建style标签并将压缩后的CSS文件内容赋值给style标签的string属性
            style_tag = soup.new_tag('style')
            style_tag.string = compressed_css_content

            # 替换link标签为style标签
            link_tag.replace_with(style_tag)
        else:
            link_tag['href'] = image_to_base64(link_tag['href'])

    # 遍历img标签
    for img_tag in img_tags:
        # 获取img标签中的src属性
        src = img_tag.get('src')
        # 如果href属性存在，则获取对应的CSS文件内容
        if src:
            img_tag['src'] = image_to_base64(img_tag['src'])



    html_content = convert_png_to_base64(str(soup))

    # 将替换后的HTML写入新的文件中
    with open(outfile, 'w', encoding='utf-8') as file:
        file.write(html_content)
          
# main(infile)