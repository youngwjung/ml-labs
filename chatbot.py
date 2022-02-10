import json
import random
import boto3
import csv
import scipy.sparse as sparse
import numpy as np



def get_session_attributes(intent_request):
    sessionState = intent_request['sessionState']
    if 'sessionAttributes' in sessionState:
        return sessionState['sessionAttributes']
    else:
        return {}


def elicit_slot(intent_request, session_attributes, message):
    return {
        'sessionState': {
            'dialogAction': {
                'slotToElicit': 'item',
                'type': 'ElicitSlot'
            },
            'intent': {
                'name': 'Shipment',
                'slots': intent_request['interpretations'][0]['intent']['slots']
            },
            'sessionAttributes': session_attributes
        },
        'messages': [ message ] if message != None else None,
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }


def close(intent_request, session_attributes, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [message],
        'sessionId': intent_request['sessionId'],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }


def shipment(intent_request):
    session_attributes = get_session_attributes(intent_request)
    
    items = ["아이폰", "매직키보드", "포카리스웨트"]
    tracking_status = ["상품준비중", "배송준비중", "배송중", "배송완료"]
    
    if intent_request['interpretations'][0]['intent']['slots']['item']:
        item = intent_request['interpretations'][0]['intent']['slots']['item']['value']['originalValue']
        
        if item in items:
            text = f"{item}는 {random.choice(tracking_status)} 입니다"
            message =  {
                'contentType': 'PlainText',
                'content': text
            }
            fulfillment_state = "Fulfilled"
            return close(intent_request, session_attributes, fulfillment_state, message)
        else:
            text = f"{item}에 대한 구매이력이 없습니다. 어떤 상품에 대한 배송상태를 확인하고 싶으신가요? \n \n"
            for item in items:
                text += f"{item} \n"
            message =  {
                'contentType': 'PlainText',
                'content': text
            }
            return elicit_slot(intent_request, session_attributes, message)
    
    else:
        text = "어떤 상품에 대한 배송상태를 확인하고 싶으신가요? \n \n"
        for item in items:
            text += f"{item} \n"
        
        message =  {
                'contentType': 'PlainText',
                'content': text
            }
        return elicit_slot(intent_request, session_attributes, message)

def serialize(data):
        js = {"instances": []}
        for row in data:
            js["instances"].append({"features": row.tolist()})
        return json.dumps(js)


def recommend(intent_request):
    session_attributes = get_session_attributes(intent_request)
    
    s3 = boto3.resource('s3')
    s3.meta.client.download_file("youngwjung", "sparse_matrix.npz", '/tmp/sparse_matrix.npz')
    
    input = sparse.load_npz('/tmp/sparse_matrix.npz')
    
    payload = input.toarray()
    body = serialize(payload)
    
    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName='ENDPOINT',
        Body=body
    )
    result = json.loads(response['Body'].read())
    predictions = [i["score"] for i in result["predictions"]]
    index_array = np.array(predictions).argsort()
    products = np.array(['B009CCXJ0A', 'B00F8K9MZQ', 'B00MYXTCGY', 'B00337D8U6',
                'B008TND0L6', 'B00J46XO9U', 'B00E5PI594', 'B004911E9M',
                'B005S1CYO6', 'B0067XVNTG'])
    top_3_recommended = np.take_along_axis(products, index_array, axis=0)[: -4 : -1]
    
    message =  {
        'contentType': 'PlainText',
        'content': ",".join(top_3_recommended)
    }
    fulfillment_state = "Fulfilled"
    return close(intent_request, session_attributes, fulfillment_state, message)


def dispatch(intent_request):
    intent_name = intent_request['sessionState']['intent']['name']
    response = None
    # Dispatch to your bot's intent handlers
    if intent_name == 'Shipment':
        return shipment(intent_request)
    elif intent_name == 'Recommend':
        return recommend(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')


def lambda_handler(event, context):
    print(event)
    response = dispatch(event)
    return response
