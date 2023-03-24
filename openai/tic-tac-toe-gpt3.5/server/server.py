from bottle import Bottle, request, response
import openai
import os
import json

openai.api_key = os.getenv("OPENAPI_KEY") 

def prompt(board, model='davinci',chat_history=[]):
    '''
    using gpt-3.5-turbo model costs $0.002/ 1k tokens.
    prompt has ~400 tokens so 3 request = ~1200 toks cost ~$0.002 

    using text-davinci-003 model costs $0.02/1k tokens.
    prompt has ~400 tokens so 3 request = ~1200 toks cost ~$0.02
    '''
    prompt = '''
    You are an intelligent agent that is an expert player of tic-tac-toe.
    Humans play X and You play O.
    The game is played on a grid that's 3 squares by 3 squares.
    Players take turns putting their marks in empty squares.
    Never play on a space that has already been played on, only play empty spaces marked by "-".
    The first player to get 3 of her marks in a row (up, down, across, or diagonally) is the winner.
    If player X have 2 of her marks in a row (up, down, across, or diagonally) you must play the next cell so he doesn't win.
    If player X have 2 of her marks separated in an empty space (up, down, across, or diagonally) you must play the empty cell so he doesn't win.
    You are playing against a human and its your turn to play, use the state in the board below to make the best possible move and remember to follow tic-tac-toe rules.

    {}  {}  {}
    {}  {}  {}
    {}  {}  {}

    Your answer must be a single json object and have the following schema: 

    {{
        "move":{{
            "row": 0,
            "col": 0
        }},
        "trashTalk": "Holly smokes this is easy!"
    }}

    rows and columns are zero indexed.
    trashTalk must be a string with a minimum of 10 characters and a maximum of 30 characters. 
    trashTalk must be funny and creative.
    trashTalk cannot be repeat nor be similar to anything in this list: {}
    '''.format(*board, chat_history)
    if model == 'davinci':
        ret = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=1000)
        return json.loads(ret.choices[0].text)
    ret = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content":"You are a expert tik tok player" },
        {"role": "user", "content": prompt},
    ])
    return json.loads(ret.choices[0].message.content)
chat_history = []
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
    res = prompt(board,model='turbo', chat_history=chat_history)
    print(res)
    chat_history.append(res["trashTalk"])
    return {'move': {'row': res["move"]["row"], 'col': res["move"]["col"]}, 'trashTalk': res["trashTalk"]}

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
