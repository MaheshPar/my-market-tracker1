# kiteapp.py
import json
import requests
import kiteconnect.exceptions as ex
import logging
from six.moves.urllib.parse import urljoin
from kiteconnect import KiteConnect, KiteTicker

log = logging.getLogger(__name__)

class KiteApp(KiteConnect):
    def __init__(self, api_key, userid, enctoken):
        self.api_key = api_key
        self.user_id = userid
        self.enctoken = enctoken
        self.headers = {
            'x-kite-version': '3', 
            'Authorization': 'enctoken {}'.format(self.enctoken)
        }
        # Initialize parent class
        KiteConnect.__init__(self, api_key=api_key)

    def kws(self):
        """Create WebSocket connection for real-time data"""
        return KiteTicker(
            api_key='kitefront', 
            access_token=self.enctoken + '&user_id=' + self.user_id, 
            root='wss://ws.kite.trade'
        )

    def _request(self, route, method, url_args=None, query_params=None, params=None, is_json=False):
        """Make an HTTP request to Kite API"""
        # Build URL
        if url_args:
            uri = self._routes[route].format(**url_args)
        else:
            uri = self._routes[route]
        
        url = urljoin(self.root, uri)
        headers = self.headers
        
        # Debug logging
        if self.debug:
            log.debug('Request: {method} {url} {params} {headers}'.format(
                method=method, url=url, params=params, headers=headers
            ))
        
        # Make the request
        try:
            if method in ['POST', 'PUT']:
                if is_json:
                    r = self.reqsession.request(
                        method, url, 
                        json=params, 
                        headers=headers,
                        verify=not self.disable_ssl,
                        timeout=self.timeout
                    )
                else:
                    r = self.reqsession.request(
                        method, url, 
                        data=params, 
                        headers=headers,
                        verify=not self.disable_ssl,
                        timeout=self.timeout
                    )
            else:  # GET, DELETE, etc.
                r = self.reqsession.request(
                    method, url, 
                    params=params, 
                    headers=headers,
                    verify=not self.disable_ssl,
                    timeout=self.timeout
                )
        except Exception as e:
            raise e
        
        # Debug logging
        if self.debug:
            log.debug('Response: {code} {content}'.format(
                code=r.status_code, content=r.content
            ))
        
        # Handle JSON response
        if 'json' in r.headers.get('content-type', ''):
            try:
                data = json.loads(r.content.decode('utf8'))
            except ValueError:
                raise ex.DataException(
                    'Couldn\'t parse the JSON response received from the server: {content}'.format(
                        content=r.content
                    )
                )
            
            # Check for error
            if data.get('error_type'):
                if self.session_expiry_hook:
                    if r.status_code == 403:
                        if data['error_type'] == 'TokenException':
                            self.session_expiry_hook()
                exp = getattr(ex, data['error_type'], ex.GeneralException)
                raise exp(data['message'], code=r.status_code)
            
            return data['data']
        
        # Handle CSV response
        if 'csv' in r.headers.get('content-type', ''):
            return r.content
        
        # Handle unknown response type
        raise ex.DataException(
            'Unknown Content-Type ({content_type}) with response: ({content})'.format(
                content_type=r.headers.get('content-type'), 
                content=r.content
            )
        )


def login_with_credentials(userid, password, twofa):
    """
    Login to Kite using username, password, and 2FA.
    Saves the enctoken to a file for later use.
    """
    reqsession = requests.Session()
    
    # Step 1: Login with userid and password
    r = reqsession.post(
        'https://kite.zerodha.com/api/login', 
        data={'user_id': userid, 'password': password}
    )
    
    # Step 2: Complete 2FA
    r = reqsession.post(
        'https://kite.zerodha.com/api/twofa', 
        data={
            'request_id': r.json()['data']['request_id'], 
            'twofa_value': twofa, 
            'user_id': r.json()['data']['user_id']
        }
    )
    
    # Get enctoken from cookies
    enctoken = r.cookies.get('enctoken')
    
    # Save to file
    with open('enctoken.txt', 'w') as wr:
        wr.write(enctoken)
    
    return enctoken


# Optional: Helper function to read enctoken from file
def get_enctoken_from_file(filename='enctoken.txt'):
    """Read enctoken from file"""
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


# Optional: Helper function to create KiteApp instance from credentials
def create_kite_app(userid=None, enctoken=None, enctoken_file='enctoken.txt'):
    """
    Create a KiteApp instance using either direct credentials or from file
    """
    if enctoken is None:
        enctoken = get_enctoken_from_file(enctoken_file)
    
    if enctoken is None:
        raise ValueError("Enctoken not provided and not found in file")
    
    if userid is None:
        # Try to get userid from enctoken or prompt
        raise ValueError("Userid is required")
    
    return KiteApp('user', userid, enctoken)


# Example usage:
if __name__ == "__main__":
    # Option 1: Login with credentials
    # enctoken = login_with_credentials('USER123', 'password', '123456')
    
    # Option 2: Use existing enctoken from file
    # kite = create_kite_app(userid='USER123')
    
    # Option 3: Direct creation
    # kite = KiteApp('user', 'USER123', 'your_enctoken_here')
    
    print("KiteApp module loaded successfully!")
