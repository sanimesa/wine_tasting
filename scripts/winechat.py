import requests
from bs4 import BeautifulSoup
import google.generativeai as palm
import pandas as pd
import time 
import openai
import json

def output_array_of_dicts_to_json(array_of_dicts, filename):
    with open(filename, 'w') as file:
        json.dump(array_of_dicts, file, indent=4)


def predict_wine_openai(tasting_note):

    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="""From the following wine tasting note, can you guess the grape varietal? 
    "{tasting_note}"
    """.format(tasting_note=tasting_note)
    )

    time.sleep(3)
    print(response)

    if 'choices' in response and len(response['choices']) > 0 and 'text' in response['choices'][0]:
        return response['choices'][0]['text']
        
    return 'N/A'


def predict_wine(tasting_note):
    palm.configure(api_key="AIzaSyAmg8Vm2hLPonlVKMW3jz3TGhiP_9Igwjs")

    defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.7,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    # 'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    }

    prompt = """From the following wine tasting note, can you guess the grape varietal? 
    "{tasting_note}"
    """.format(tasting_note=tasting_note)

    response = palm.generate_text(
    **defaults,
    prompt=prompt
    )

    time.sleep(3)

    print(response.result)

    return response.result

cabernet_wines = []
merlot_wines = []
chardonnay_wines = []

def extractWine(page): 
    segment_html = '''
    <div class="md:text-3-xl">
        <h3 class="mb-8">
            <a href="/wine/wine-detail/id/1298120/name/garnacha-campo-de-borja-2020" class="text-gray-darkest">BODEGAS ALTO MONCAYO</a>
        </h3>
        <h4 class="mb-8">
            <a href="/wine/wine-detail/id/1298120/name/garnacha-campo-de-borja-2020" class="text-gray-darkest">Garnacha Campo De Borja 2020</a>
        </h4>
        <p>
            $47
        </p>
        <p>
            A sleek, harmonious red, with a rich undertow of sweet smoke, fig jam and mocha notes, plus generous flavors of blackberry paste, crushed black cherry and grilled thyme. Long and silky, with lots of fragrant spices on the finish. Drink now through 2030. 4,550 cases made, 1,800 cases imported.
            <em>&mdash;Alison Napjus </em>
        </p>
    </div>
    '''

    # Create a BeautifulSoup object to parse the segment HTML
    soup = BeautifulSoup(page, 'html.parser')
    segments = soup.find_all('div', class_='md:text-3-xl')
    # print(segments)

    for segment in segments: 

        # Extract the vineyard name
        vineyard = segment.select_one('h3 > a').text.strip()

        # Extract the wine name
        wine_name = segment.select_one('h4 > a').text.strip()

        # Extract the price
        price = segment.select_one('p:nth-child(3)').text.strip()

        # Extract the tasting notes
        tasting_notes = segment.select_one('p:nth-child(4)').text.strip()

        # Output the extracted information
        # print("Vineyard:", vineyard)
        # print("Wine:", wine_name)
        # print("Price:", price)
        # print("Tasting Notes:", tasting_notes)

        notes = tasting_notes
        if 'Drink now through' in tasting_notes:
            notes = tasting_notes.split('Drink now through')[0]
        if 'To be released' in tasting_notes:
            notes = tasting_notes.split('To be released')[0]
        if 'Drink now' in tasting_notes:
            notes = tasting_notes.split('Drink now')[0]
        if 'Best from' in tasting_notes:
            notes = tasting_notes.split('Best from')[0]

        wine = {'vineyard': vineyard, 'wine_name': wine_name, 'price': price, 'tasting_note': notes}

        if 'cabernet' in str(wine_name).lower():
            cabernet_wines.append(wine)

        if 'chardonnay' in str(wine_name).lower():
            chardonnay_wines.append(wine)

        if 'merlot' in str(wine_name).lower():
            print('merlot:', wine)
            merlot_wines.append(wine)

        # break

def scrape_wine_reviews():
    # Range of URLs to fetch
    start_page = 2
    end_page = 250 

    # Iterate over the page numbers and fetch the corresponding URLs
    for n in range(start_page, end_page + 1):
        url = f'https://www.winespectator.com/dailypicks/more_than_40/page/{n}'

        # Send a GET request to the webpage
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the content of the page
            page_content = response.content

            # Invoke the extractValues function with the page content
            extractWine(page_content)
        else:
            print(f"Failed to fetch page {n}. Status code: {response.status_code}")


    print('cabernet wines:', len(cabernet_wines), cabernet_wines)
    print('chardonnay wines:', len(chardonnay_wines), chardonnay_wines)
    print('merlot wines:', len(merlot_wines), merlot_wines)

    # Output the array of dictionaries to a JSON file
    output_array_of_dicts_to_json(cabernet_wines, 'c:/temp/cabernet_output_1.json')
    output_array_of_dicts_to_json(chardonnay_wines, 'c:/temp/chardonnay_output_1.json')
    output_array_of_dicts_to_json(merlot_wines, 'c:/temp/merlot_output_1.json')

def predict_wine_from_file():
    with open("c:/temp/chardonnay_output.json", "r") as file:
        content = file.read()
        wines  = json.loads(content)
        print(file.name, 'contains ', len(wines), ' reviews')

        for wine in wines:
            print(wine)
            prediction = predict_wine(wine['tasting_note'])
            print(type(wine))
            wine['prediction_palm'] = prediction
            prediction = predict_wine_openai(wine['tasting_note'])
            wine['prediction_openai'] = prediction
            
    with open('c:/temp/chardonnay_output_w_predictions_both.json', 'w') as file:
        json.dump(wines, file, indent=4)

def predict_wine_from_fake_reviews():
    with open("c:/temp/chatgpt_generated_fake_cabernet_tasting_notes.json", "r") as file:
        content = file.read()
        wines  = json.loads(content)
        print(file.name, 'contains ', len(wines), ' reviews')

        for wine in wines:
            print(wine)
            prediction = predict_wine(wine['note'])
            print(type(wine))
            wine['prediction_palm'] = prediction
            prediction = predict_wine_openai(wine['note'])
            wine['prediction_openai'] = prediction
            
    with open('c:/temp/chatgpt_generated_fake_cabernet_tasting_notes_w_predictions_both.json', 'w') as file:
        json.dump(wines, file, indent=4)


def main():
    scrape_wine_reviews()
    # predict_wine_from_fake_reviews()
    # predict_wine_from_file()

    # for wine in cabernet_wines:
    #     prediction = predict_wine(wine['tasting_note'])
    #     print(f"wine name: {wine['wine_name']} prediction: {prediction}")
    #     wine['prediction'] = prediction


    # df = pd.DataFrame.from_dict(cabernet_wines)
    # html = df.to_html( index=False, classes='stocktable', table_id='table1')

    # with open("c:/temp/wine_table.html", "w") as file:
    #     file.write(html)

if __name__ == '__main__':
    main()