from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from etf_scraper import ETFScraper


def get_ticker_holdings(fund_ticker,holdings_date = None, print_output=False):
    """
    Retrieves the holdings information for a given ETF ticker using ETFScraper.
    
    :param fund_ticker: str
    :param holdings_date: str or None (None to query the latest holdings)
    :param print_output: True if you want to print header rows of the holdings info
    :return: pandas DataFrame containing the holdings information
    """
    # initialization & setup
    etf_scraper = ETFScraper()
    
    try:
        holdings_df = etf_scraper.query_holdings(fund_ticker, holdings_date)
    except Exception as e:
        print(f"Error retrieving holdings for {fund_ticker}: {e}")
        return None
    print(f'Holdings Info Retrieved for {fund_ticker}')


    if print_output:
        # log the holdings information
        print(f"Holdings for {fund_ticker}:")
        print(holdings_df.head(5))  # print first xx rows
        print("\n")
    
    return holdings_df


def extract_subtable(my_soup, subtable_key):
    """
    Extracts a subtable from the provided BeautifulSoup 
    object(from www.justetf.com) based on the given subtable key (see header_dict).
    
    :param my_soup: a BeautifulSoup object containing the HTML content
    :param subtable_key: a value in header_dict, e.g., "top-holdings", "countries", "sectors"
    :return: a pandas DataFrame with columns "Name" and "Value" which contains the extracted info
    """
    header_0 = "etf-holdings"

    def extract_name_from_cell(cell):
        a = cell.find("a")
        if a and a.get_text(strip=True):
            return a.get_text(strip=True)
        # look for element with the data-testid (don't search for a <td> inside a <td>)
        el = cell.find(
            attrs={"data-testid": f"tl_{header_0}_{subtable_key}_value_name"}
        )
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
        # try common inline tags
        for tag in ("span", "div", "p", "strong"):
            t = cell.find(tag)
            if t and t.get_text(strip=True):
                return t.get_text(strip=True)
        # final fallback: plain text of the td
        txt = cell.get_text(separator=" ", strip=True)
        return txt if txt else "N/A"

    table_header = f"{header_0}_{subtable_key}_table"
    row_header = f"{header_0}_{subtable_key}_row"
    table = my_soup.find("table", {"data-testid": table_header})

    data = []

    if table:
        rows = table.find_all("tr", {"data-testid": row_header})
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                # Extract name
                name = extract_name_from_cell(cells[0])
                # Extract value
                value_span = cells[1].find(
                    "span",
                    {"data-testid": f"tl_{header_0}_{subtable_key}_value_percentage"},
                )
                value = value_span.text.strip() if value_span else "N/A"
                data.append({"Name": name, "Value": value})

        return pd.DataFrame(data)
