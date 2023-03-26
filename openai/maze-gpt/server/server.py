from bottle import Bottle, request, response
import openai
import os
import json

openai.api_key = os.getenv("OPENAPI_KEY") 

def prompt(board, model='gpt-3.5-turbo'):
    '''
    using gpt-3.5-turbo model costs $0.002/ 1k tokens.
    using text-davinci-003 model costs $0.02/1k tokens.
    '''
    prompt = '''
    You are playing a game, the game consist of the following rules:
    You are in a {} grid, each cell can have 4 different states: 'X', 'O', 'B' or '-' 
    There is only one cell marked as X, this is the entry point where you start playing.
    There is only one cell marked as 0, this is the destination where you must end playing.
    cell marked as B are blocked spaces, you cannot under any circumstance move into a space that is blocked.
    cell marked as - are empty spaces, where you can freely move. However, you can move right, left, down, up, but only one at the time.
    you cannot jump between cells, you must move one cell at the time.
    Your objective is to reach the destination cell in the least amount of moves possible.

    {} 

    You must only respond with an valid json object that have the following schema: 

    {{
        "path":  [[row, col]]
    }}

    The only key in the object will be "path" which value is a list of tuple, each tuple represents a move, the first element of the tuple is the row and the second element is the column.
    the path must not include the entry point X, and it must not include the destination point O.
    Avoid appending any extra key into the JSON object, otherwise the game will not work.
    Do not under any circumstance reponse with anything else than the JSON object with the "path" key.
    Avoid replying with an explanation at all cost. 
    The grid is 0 indexed.

    '''.format(board, f"{len(board)}X{len(board[0])}")
    if model == 'text-davinci-003':
        ret = openai.Completion.create(model=model, prompt=prompt, temperature=0, max_tokens=1000)
        return json.loads(ret.choices[0].text)
    ret = openai.ChatCompletion.create(model=model,
    messages=[
        {"role": "system", "content":"You are an expert maze solver" },
        {"role": "user", "content": prompt},
    ])
    print(ret)
    return json.loads(ret.choices[0].message.content)

app = Bottle()
@app.route('/', method=['OPTIONS','POST'])
def make_move():
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
    if request.method == 'OPTIONS':
        return {}
    data = request.json
    board = data['board']
    res = prompt(board)
    return {'path':res['path']} 

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
