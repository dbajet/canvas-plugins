from datetime import datetime

from canvas_sdk.clients.aws.structures.aws_s3_item import AwsS3Item
from canvas_sdk.tests.conftest import is_namedtuple


def test_class() -> None:
    """Test AwsS3Item is a NamedTuple with correct fields and types."""
    assert is_namedtuple(
        AwsS3Item,
        {
            "key": str,
            "size": int,
            "last_modified": datetime,
        },
    )
