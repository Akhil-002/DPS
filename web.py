import requests
import xml.etree.ElementTree as ET

url = f'https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=fungalinfection+medicines'
response = requests.get(url)
xml_response = response.text

root = ET.fromstring(xml_response)

# Extract the content within the FullSummary tag
full_summary_content = root.find('.//content[@name="FullSummary"]').text

print(f"Full Summary Content:\n{full_summary_content}")