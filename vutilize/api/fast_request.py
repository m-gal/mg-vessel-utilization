"""
    Check API with FastAPI

    1. Run fast_api.py
    2. Run from Terminal:
        ./vessel_utilize => [ Open in Integrated Terminal ]
            >>> python api/fast_request.py
        or ./vessel_utilize/api => [ Open in Integrated Terminal ]
            >>> python fast_request.py

    Response body must be:
    b'{"teu_esimated":561.0,"vu_esimated":0.4656}'

    @author: mikhail.galkin
"""

#%%
if __name__ == "__main__":
    import requests

    data = {"dim_a": 154, "dim_b": 49, "dim_c": 3, "dim_d": 23, "draught": 9.7}
    response = requests.post("http://127.0.0.1:8000/utilize/predict", json=data)
    print(response.content)

#%%
